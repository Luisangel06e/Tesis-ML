#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semana 7 | Clasificación supervisada y gráficas para BreaKHis (40X/100X).

Este programa continúa el trabajo de la semana 6. Lee las características
numéricas del CSV y compara tres clasificadores: SVM, Random Forest y XGBoost.
Al terminar deja las métricas en tablas y genera las gráficas para el informe.

El punto más importante de la evaluación es NO mezclar fotografías de un mismo
paciente entre entrenamiento y evaluación: varias imágenes del mismo caso
pueden ser muy parecidas y dar una idea exagerada de la precisión del modelo.

Comando de ejemplo en Windows, dentro de la carpeta Estancia:
    python "Modelo Supervisados.py" --csv "resultados_punto_6/caracteristicas_por_imagen.csv" --output resultados_semana7 --folds 5

Las etiquetas numéricas son 0 = benigna y 1 = maligna.
Estas métricas describen un experimento de investigación, no un diagnóstico.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Guardar imágenes sin abrir ventanas, también en Windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    xgboost = None
    XGBClassifier = None


# Evitamos usar nombres de archivos, etiquetas o valores geométricos que
# identificamos como inconsistentes en una versión anterior de la semana 6.
# Los nombres, identificadores y etiquetas NO son características visuales.
# También dejamos fuera las geometrías agregadas que habían dado problemas
# en versiones anteriores; se podrán reincorporar tras validar el recálculo.
EXCLUIR = {
    "archivo", "magnificacion", "clase", "subtipo", "paciente", "metodo",
    "tipo_mascara", "ancho", "alto", "particion",
    "area_media_region_px", "circularidad_media", "solidez_media",
}


def localizar_csv(ruta: str | None) -> Path:
    """Encuentra el CSV en las carpetas que hemos usado durante la estancia."""
    if ruta:
        archivo = Path(ruta).expanduser()
        if not archivo.is_file():
            raise FileNotFoundError(f"No existe el CSV indicado: {archivo.resolve()}")
        return archivo

    carpetas = [Path.cwd(), Path(__file__).resolve().parent]
    opciones = []
    for carpeta in carpetas:
        opciones.extend([
            carpeta / "resultados_punto_6_corregidos" / "caracteristicas_por_imagen.csv",
            carpeta / "resultados_punto_6" / "caracteristicas_por_imagen.csv",
            carpeta / "caracteristicas_por_imagen.csv",
        ])
    for opcion in opciones:
        if opcion.is_file():
            return opcion
    raise FileNotFoundError(
        "No encontré caracteristicas_por_imagen.csv. Ejecuta el script con "
        "--csv \"RUTA_COMPLETA_AL_CSV\"."
    )


def cargar_datos(ruta: Path, metodo: str, mascara: str):
    """Obtiene una sola versión de cada imagen y sus columnas numéricas.

    Si mezcláramos las versiones none, Reinhard y Macenko en la misma tabla,
    la misma fotografía terminaría contándose varias veces.
    """
    """Carga una sola condición experimental para no duplicar imágenes."""
    df = pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)
    columnas = {"archivo", "magnificacion", "clase", "paciente", "metodo", "tipo_mascara"}
    faltantes = sorted(columnas - set(df.columns))
    if faltantes:
        raise ValueError(f"El CSV no contiene estas columnas necesarias: {faltantes}")

    df = df.loc[
        (df["metodo"] == metodo)
        & (df["tipo_mascara"] == mascara)
        & (df["magnificacion"].isin(["40X", "100X"]))
        & (df["clase"].isin(["benigna", "maligna"]))
    ].copy()
    if df.empty:
        raise ValueError(f"No hay datos para método={metodo}, máscara={mascara}.")

    # Un ID de paciente ausente impediría comprobar que no hay fuga de datos.
    df = df.dropna(subset=["archivo", "paciente"]).copy()
    df["paciente"] = df["paciente"].astype(str).str.strip()
    df = df.loc[~df["paciente"].isin(["", "nan", "None", "no_identificado"])].copy()
    if df.empty:
        raise ValueError("Los datos seleccionados no tienen identificadores de paciente válidos.")
    # La combinación de una única normalización y máscara debe dejar una
    # sola fila por imagen. Si no ocurre, detenemos el experimento.
    if df.duplicated(["archivo"]).any():
        raise ValueError("Hay imágenes repetidas en la misma condición experimental.")
    if df.groupby("paciente")["clase"].nunique().gt(1).any():
        raise ValueError("Un mismo identificador de paciente aparece con ambas clases.")

    features = [
        c for c in df.columns
        if c not in EXCLUIR and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    if X.empty:
        raise ValueError("No quedan características numéricas para entrenar.")
    return df, X, X.columns.tolist()


def construir_modelos(y_entrenamiento: np.ndarray, semilla: int):
    """Crea modelos nuevos para cada partición de entrenamiento.

    La imputación y el escalado van DENTRO de Pipeline; así sus parámetros
    se calculan solo con el fold de entrenamiento. Hacerlos antes de separar
    los folds filtraría información del conjunto evaluado.
    """
    """Crea modelos nuevos en cada fold: no reutilizamos modelos ya entrenados."""
    positivas = max(1, int(np.sum(y_entrenamiento == 1)))
    negativas = int(np.sum(y_entrenamiento == 0))

    modelos = {
        "SVM": Pipeline([
            ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("escalar", StandardScaler()),
            ("modelo", SVC(C=1.0, kernel="rbf", class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("modelo", RandomForestClassifier(
                n_estimators=180, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", random_state=semilla, n_jobs=-1,
            )),
        ]),
    }
    if XGBClassifier is not None:
        modelos["XGBoost"] = Pipeline([
            ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("modelo", XGBClassifier(
                n_estimators=220, max_depth=3, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", scale_pos_weight=negativas / positivas,
                random_state=semilla, n_jobs=4,
            )),
        ])
    return modelos


def nombre_archivo(texto: str) -> str:
    """Construye nombres sin espacios ni símbolos problemáticos."""
    return re.sub(r"[^A-Za-z0-9_+\-]", "_", texto.replace(" ", "_"))


def guardar_figura(fig, ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# GRÁFICAS PARA EL INFORME
# -------------------------------------------------------------------------

def grafica_confusion(reales, predicciones, modelo: str, carpeta: Path):
    """Suma predicciones fuera de fold; cada imagen debe contarse una vez."""
    matriz = confusion_matrix(reales, predicciones, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=["Benigna", "Maligna"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matriz de confusión · {modelo}\nPredicciones fuera de fold")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    guardar_figura(fig, carpeta / f"{nombre_archivo(modelo)}_matriz_confusion.png")


def grafica_roc_folds(curvas: list[tuple[np.ndarray, np.ndarray, float]],
                       modelo: str, carpeta: Path):
    """Grafica ROC por fold; evita unir puntuaciones con escalas distintas.

    Especialmente para SVM, las puntuaciones de folds diferentes no tienen
    por qué estar calibradas en una misma escala. Por eso graficamos cada
    curva y reportamos el promedio de sus AUC, no un AUC global mezclado.
    """
    if not curvas:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    cuadricula = np.linspace(0, 1, 101)
    interpolaciones = []
    for indice, (fpr, tpr, auc) in enumerate(curvas, start=1):
        ax.plot(fpr, tpr, linewidth=1.2, alpha=0.55,
                label=f"Fold {indice}: AUC {auc:.3f}")
        interp = np.interp(cuadricula, fpr, tpr)
        interp[0] = 0.0
        interpolaciones.append(interp)
    tpr_media = np.mean(interpolaciones, axis=0)
    tpr_media[-1] = 1.0
    auc_media = float(np.mean([c[2] for c in curvas]))
    ax.plot(cuadricula, tpr_media, linewidth=2.8,
            label=f"Curva media (AUC medio de folds: {auc_media:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Azar")
    ax.set(xlabel="Tasa de falsos positivos (1 - especificidad)",
           ylabel="Sensibilidad", title=f"Curvas ROC · {modelo}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    guardar_figura(fig, carpeta / f"{nombre_archivo(modelo)}_curvas_ROC.png")


def grafica_importancias(importancias_folds: list[np.ndarray], features: list[str],
                          modelo: str, carpeta: Path, top: int = 15):
    """Importancia media de los árboles ajustados en los folds de entrenamiento.

    Es una importancia interna de cada modelo, no una medida de causalidad
    o relevancia clínica. El SVM con kernel RBF no expone estas importancias.
    """
    if not importancias_folds:
        return None
    imp = np.vstack(importancias_folds)
    prom = imp.mean(axis=0)
    desv = imp.std(axis=0)
    indices = np.argsort(prom)[-top:]
    fig, ax = plt.subplots(figsize=(9, max(5, len(indices) * 0.34 + 1.2)))
    posiciones = np.arange(len(indices))
    ax.barh(posiciones, prom[indices], xerr=desv[indices], capsize=2)
    ax.set_yticks(posiciones, labels=[features[i] for i in indices])
    ax.set_xlabel("Importancia media (± desviación entre folds)")
    ax.set_title(f"Variables destacadas · {modelo}")
    fig.tight_layout()
    guardar_figura(fig, carpeta / f"{nombre_archivo(modelo)}_importancia_variables.png")
    return pd.DataFrame({
        "modelo": modelo,
        "variable": features,
        "importancia_media_folds": prom,
        "importancia_desv_folds": desv,
    })


def grafica_comparacion(resumen: pd.DataFrame, carpeta: Path, configuracion: str):
    """Pone las métricas de todos los modelos lado a lado.

    No debemos elegir un modelo solo por accuracy: en este problema también
    importan los falsos negativos, la sensibilidad y la especificidad.
    """
    """Compara los tres modelos usando métricas entre 0 y 1."""
    if resumen.empty:
        return
    columnas = [
        ("accuracy_oof", "Exactitud"),
        ("precision_oof", "Precisión"),
        ("sensibilidad_oof", "Sensibilidad"),
        ("especificidad_oof", "Especificidad"),
        ("f1_oof", "F1"),
        ("auc_media_folds", "AUC medio"),
    ]
    eje_x = np.arange(len(resumen))
    ancho = 0.12
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for i, (col, etiqueta) in enumerate(columnas):
        pos = eje_x + (i - (len(columnas) - 1) / 2) * ancho
        ax.bar(pos, resumen[col].to_numpy(dtype=float), width=ancho, label=etiqueta)
    ax.set_xticks(eje_x, labels=resumen["modelo"].to_list())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title(f"Comparación de modelos · {configuracion}")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    guardar_figura(fig, carpeta / "comparacion_metricas.png")


# -------------------------------------------------------------------------
# ENTRENAMIENTO Y VALIDACIÓN CRUZADA
# -------------------------------------------------------------------------

def main():
    """Entrena, evalúa y guarda todas las tablas y figuras de la semana 7."""
    parser = argparse.ArgumentParser(
        description="Semana 7: clasificación supervisada, validación por paciente y gráficas"
    )
    parser.add_argument("--csv", default=None, help="Ruta al CSV de características de la semana 6")
    parser.add_argument("--output", default="resultados_semana7", help="Carpeta donde se guardará todo")
    parser.add_argument("--method", choices=["none", "reinhard", "macenko", "vahadane"], default="none")
    parser.add_argument("--mask", choices=["hsv", "lab"], default="hsv")
    parser.add_argument("--folds", type=int, default=5, help="Número de folds agrupados por paciente")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", choices=["train", "all"], default="train",
                        help="train respeta la reserva de test si existe columna particion")
    parser.add_argument("--magnification", choices=["40X", "100X", "ambas", "all"], default="all")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds debe ser 2 o mayor.")

    csv = localizar_csv(args.csv)
    df, X, features = cargar_datos(csv, args.method, args.mask)

    if "particion" in df.columns and args.subset == "train":
        # Algunos CSV antiguos traen la columna vacía. Eso NO equivale a
        # disponer de una partición de entrenamiento comprobada.
        seleccion = df["particion"].astype(str).str.strip().str.lower().eq("train")
        if not seleccion.any():
            valores = set(df["particion"].fillna("").astype(str).str.strip().str.lower())
            if valores <= {"", "nan", "none"}:
                print("ADVERTENCIA: la columna particion está vacía. "
                      "Se hará validación exploratoria por paciente, sin test reservado.")
                seleccion = pd.Series(True, index=df.index)
            else:
                raise ValueError("La columna particion tiene valores, pero ninguna fila train. "
                                 "Revisa cómo se construyeron las particiones.")
        df = df.loc[seleccion].copy()
        X = X.loc[df.index]
        print("Usando solamente train. La partición de test queda reservada.")
    elif "particion" not in df.columns:
        print("ADVERTENCIA: el CSV no distingue train/test. Esta validación es EXPLORATORIA.")
    else:
        print("ADVERTENCIA: --subset all incluye cualquier test reservado. No usar para selección final.")
    if args.method != "none":
        print("AVISO: si la referencia de tinción se escogió antes de definir cada fold, "
              "la comparación es exploratoria. Para validación estricta se ajusta "
              "la referencia dentro de cada fold.")
    if XGBClassifier is None:
        print("AVISO: XGBoost no está instalado. Instala con: python -m pip install xgboost")

    salida = Path(args.output)
    salida.mkdir(parents=True, exist_ok=True)
    registros_fold, registros_oof, resumen_total = [], [], []
    importancias_total = []
    configuraciones = (["40X", "100X", "40X+100X"] if args.magnification == "all" else
                      ["40X+100X" if args.magnification == "ambas" else args.magnification])

    # Analizamos 40X, 100X y ambas juntas. Son tres comparaciones distintas;
    # una diferencia entre ellas puede deberse a la escala o a la muestra.
    for configuracion in configuraciones:
        sel = (df["magnificacion"].isin(["40X", "100X"])
               if configuracion == "40X+100X"
               else df["magnificacion"].eq(configuracion))
        sub = df.loc[sel].copy()
        if sub.empty:
            print(f"Sin imágenes para {configuracion}; se omite.")
            continue
        xx = X.loc[sub.index]
        y = (sub["clase"] == "maligna").astype(int).to_numpy()
        grupos = sub["paciente"].astype(str).to_numpy()
        n_pacientes = sub.groupby("clase")["paciente"].nunique()
        if any(n_pacientes.get(clase, 0) < args.folds for clase in ["benigna", "maligna"]):
            raise ValueError(f"{configuracion}: faltan pacientes por clase para {args.folds} folds: "
                             f"{n_pacientes.to_dict()}. Prueba con --folds 3.")

        # StratifiedGroupKFold intenta mantener el balance de clases, pero
        # respeta principalmente que ningún paciente se comparta entre folds.
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        particiones = list(cv.split(xx, y, grupos))
        for train, test in particiones:
            if set(grupos[train]) & set(grupos[test]):
                raise RuntimeError("Fuga de información: un paciente está en train y test.")
            if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                raise ValueError(f"En {configuracion} hay un fold con una sola clase. "
                                 "Cambia --folds o --seed y vuelve a probar.")

        carpeta_graficas = salida / "graficas" / configuracion
        carpeta_graficas.mkdir(parents=True, exist_ok=True)
        nombres = list(construir_modelos(y, args.seed).keys())
        # OOF = out of fold: cada imagen recibe una predicción de un modelo
        # que NO fue entrenado con ese paciente. Con ellas hacemos las
        # matrices de confusión y las métricas finales por fotografía.
        pred_oof = {nombre: np.full(len(sub), -1, dtype=int) for nombre in nombres}
        cobertura = {nombre: np.zeros(len(sub), dtype=int) for nombre in nombres}
        rocs = {nombre: [] for nombre in nombres}
        importancias = {nombre: [] for nombre in nombres}
        filas_resumen = []

        print(f"\n{configuracion}: {len(sub)} imágenes de {len(set(grupos))} pacientes")
        for fold, (train, test) in enumerate(particiones, start=1):
            print(f"  Fold {fold}/{args.folds}: train={len(train)}, evaluación={len(test)}")
            modelos = construir_modelos(y[train], args.seed + fold)
            for nombre, modelo in modelos.items():
                modelo.fit(xx.iloc[train], y[train])
                pred = modelo.predict(xx.iloc[test]).astype(int)
                # Para ROC usamos un score continuo. En SVM es la distancia
                # firmada a la frontera; en los árboles, probabilidad estimada.
                if nombre == "SVM":
                    puntuaciones = modelo.decision_function(xx.iloc[test])
                else:
                    puntuaciones = modelo.predict_proba(xx.iloc[test])[:, 1]
                pred_oof[nombre][test] = pred
                cobertura[nombre][test] += 1

                tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[0, 1]).ravel()
                fpr, tpr, _ = roc_curve(y[test], puntuaciones)
                auc_fold = roc_auc_score(y[test], puntuaciones)
                rocs[nombre].append((fpr, tpr, float(auc_fold)))
                # Guardamos las métricas de cada fold y sus cuatro conteos.
                # FN = lesión maligna predicha benigna; exige especial atención.
                registros_fold.append({
                    "configuracion": configuracion, "modelo": nombre, "fold": fold,
                    "n_test": len(test), "pacientes_test": len(set(grupos[test])),
                    "accuracy": accuracy_score(y[test], pred),
                    "precision": precision_score(y[test], pred, zero_division=0),
                    "sensibilidad": recall_score(y[test], pred, zero_division=0),
                    "especificidad": tn / (tn + fp) if tn + fp else np.nan,
                    "f1": f1_score(y[test], pred, zero_division=0),
                    "auc_roc": auc_fold,
                    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                })

                for posicion, valor, clase_predicha in zip(test, puntuaciones, pred):
                    r = sub.iloc[posicion]
                    registros_oof.append({
                        "archivo": r["archivo"], "paciente": r["paciente"],
                        "magnificacion": r["magnificacion"], "clase_real": r["clase"],
                        "configuracion": configuracion, "modelo": nombre, "fold": fold,
                        "prediccion": int(clase_predicha), "score": float(valor),
                    })

                clasificador = modelo.named_steps["modelo"]
                if hasattr(clasificador, "feature_importances_"):
                    importancias[nombre].append(clasificador.feature_importances_)

        # Cerrados los folds, cada imagen debe haberse evaluado exactamente
        # una vez. Si no, la comparación estaría mal construida.
        for nombre in nombres:
            if not np.all(cobertura[nombre] == 1):
                raise RuntimeError("Hay imágenes evaluadas cero o más de una vez.")
            pred = pred_oof[nombre]
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            pliegues = pd.DataFrame([
                fila for fila in registros_fold
                if fila["configuracion"] == configuracion and fila["modelo"] == nombre
            ])
            fila = {
                "configuracion": configuracion, "modelo": nombre,
                "n_imagenes": len(sub), "n_pacientes": len(set(grupos)),
                "accuracy_oof": accuracy_score(y, pred),
                "sensibilidad_oof": recall_score(y, pred, zero_division=0),
                "especificidad_oof": tn / (tn + fp) if tn + fp else np.nan,
                "precision_oof": precision_score(y, pred, zero_division=0),
                "f1_oof": f1_score(y, pred, zero_division=0),
                "auc_media_folds": pliegues["auc_roc"].mean(),
                "auc_desv_folds": pliegues["auc_roc"].std(),
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            }
            resumen_total.append(fila)
            filas_resumen.append(fila)
            grafica_confusion(y, pred, nombre, carpeta_graficas)
            grafica_roc_folds(rocs[nombre], nombre, carpeta_graficas)
            tabla = grafica_importancias(importancias[nombre], features, nombre, carpeta_graficas)
            if tabla is not None:
                tabla.insert(0, "configuracion", configuracion)
                importancias_total.append(tabla)
        grafica_comparacion(pd.DataFrame(filas_resumen), carpeta_graficas, configuracion)

    if not resumen_total:
        raise ValueError("No se encontraron imágenes para las magnificaciones seleccionadas.")

    # Las tablas permiten reproducir las cifras de las gráficas sin copiar
    # manualmente números de la pantalla.
    pd.DataFrame(registros_fold).to_csv(salida / "metricas_por_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(resumen_total).to_csv(salida / "resumen_modelos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registros_oof).to_csv(salida / "predicciones_fuera_de_fold.csv", index=False, encoding="utf-8-sig")
    if importancias_total:
        pd.concat(importancias_total, ignore_index=True).to_csv(
            salida / "importancia_variables_folds.csv", index=False, encoding="utf-8-sig"
        )
    (salida / "metadatos_ejecucion.json").write_text(json.dumps({
        "csv": str(csv.resolve()), "metodo": args.method, "mascara": args.mask,
        "magnificacion": args.magnification, "folds": args.folds,
        "semilla": args.seed, "subset": args.subset,
        "n_imagenes_tras_filtros": len(df), "n_pacientes": int(df["paciente"].nunique()),
        "variables_utilizadas": features,
        "excluidas": sorted(EXCLUIR),
        "version_sklearn": sklearn_version,
        "version_xgboost": getattr(xgboost, "__version__", None),
        "observacion": (
            "Evaluación exploratoria por paciente. Las métricas son por imagen; "
            "la curva ROC usa scores de cada fold por separado. "
            "Las importancias de árboles no implican causalidad ni utilidad clínica."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRESUMEN DE LA EVALUACIÓN:")
    print(pd.DataFrame(resumen_total).round(4).to_string(index=False))
    print(f"\nGráficas y CSV guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
