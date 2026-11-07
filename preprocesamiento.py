#!/usr/bin/env python3
"""Semanas 1 a 5 | Preparación de imágenes BreaKHis de 40X y 100X.

Este es el primer programa del proyecto. Revisa las imágenes originales,
identifica su clase y paciente, crea las particiones y prepara distintas
versiones de color para comparar su comportamiento.

¿Qué se genera?
  - Un inventario de los archivos y conteos por clase y magnificación.
  - Imágenes originales o normalizadas con Reinhard, Macenko y, si está
    instalado, Vahadane.
  - Canales HSV/Lab y máscaras antes y después de la limpieza.

Ejemplo rápido desde la carpeta Estancia:
    python analisis_breakhis.py --dataset "RUTA_A_BREaKHis" --output resultados --limit 10

Para procesar todo el conjunto se cambia --limit 10 por --limit 0.
Aquí NO se diagnostica cáncer: las máscaras todavía no han sido validadas.
"""
from __future__ import annotations
import argparse
import re
import warnings
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Admitimos los formatos de imagen más habituales. El resto de archivos
# (CSV, documentación, etc.) no entra al inventario.
EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def metadata(path: Path):
    """Obtiene lo que BreaKHis codifica en el nombre del archivo.

    Por ejemplo, en SOB_B_A-14-22549AB-40-001.png:
    B indica lesión benigna, A es el subtipo, 14-22549AB identifica
    al paciente y 40 indica la magnificación. Esta última parte nos permite
    dejar fuera 200X y 400X sin tener que mover manualmente los archivos.
    """
    name = path.stem
    full = str(path).replace('\\', '/')
    # BreaKHis: magnificación en el nombre SOB_...-40-001.png, o en carpeta 40X.
    m = re.search(r'-(40|100)-\d+$', name, flags=re.I)
    if not m:
        m = re.search(r'(?<!\d)(40|100)\s*[xX](?!\d)', full)
    if not m:
        return None
    mag = int(m.group(1))
    cls = None
    subtype = None
    n = re.search(r'SOB_([BM])_([A-Za-z]+)-', name, re.I)
    if n:
        cls = {'B': 'benigna', 'M': 'maligna'}[n.group(1).upper()]
        subtype = n.group(2).upper()
    else:
        low = full.lower()
        if re.search(r'(^|[/_])benign(a|o)?([/_]|$)', low):
            cls = 'benigna'
        elif re.search(r'(^|[/_])malignan?t?([/_]|$)', low):
            cls = 'maligna'
    p = re.search(r'SOB_[BM]_[A-Za-z]+-(.+?)-(?:40|100)-\d+$', name, re.I)
    patient = p.group(1) if p else ''
    return {'ruta': str(path.resolve()), 'archivo': path.name,
            'magnificacion': f'{mag}X', 'clase': cls or 'no_identificada',
            'subtipo': subtype or '', 'paciente': patient}


def read_rgb(path):
    """Carga una imagen en RGB y admite rutas con acentos en Windows.

    OpenCV suele trabajar en BGR. Convertimos a RGB porque los métodos de
    normalización y las comparaciones visuales emplean ese orden de canales.
    """
    # imdecode admite algunas rutas con caracteres Unicode en Windows.
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Imagen ilegible: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb):
    """Guarda la imagen procesada en PNG sin cambiar el nombre original de base."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f'No se pudo guardar: {path}')
    buf.tofile(str(path))


# -------------------------------------------------------------------------
# NORMALIZACIÓN DE TINCIÓN
# -------------------------------------------------------------------------
# Los métodos siguientes trabajan con una imagen de referencia. No hay una
# única referencia universal: conviene revisar que represente bien la tinción
# de las muestras y anotar cuál se eligió.


def od(rgb):
    """Convierte intensidad RGB a densidad óptica aproximada.

    En las muestras H&E, más absorción de luz suele implicar una mayor
    concentración aparente de tinción. Evitamos log(0) reemplazando 0 por 1.
    """
    return -np.log(np.maximum(rgb.astype(np.float64), 1.0) / 255.0)


def estimate_stain_matrix(rgb, alpha=1.0, beta=0.15):
    """Estima dos direcciones dominantes de tinción como hace Macenko.

    beta ayuda a descartar píxeles casi blancos; alpha descarta los extremos
    de la distribución angular. Después usamos una descomposición espectral
    para aproximar las direcciones de hematoxilina y eosina.
    """
    x = od(rgb).reshape(-1, 3)
    x = x[np.all(x > beta, axis=1)]
    if len(x) < 50:
        raise ValueError('Insuficientes píxeles teñidos para Macenko')
    _, vecs = np.linalg.eigh(np.cov(x, rowvar=False))
    plane = vecs[:, -2:]
    projection = x @ plane
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    amin, amax = np.percentile(angles, [alpha, 100 - alpha])
    v1 = plane @ np.array([np.cos(amin), np.sin(amin)])
    v2 = plane @ np.array([np.cos(amax), np.sin(amax)])
    if v1[0] < v2[0]:
        H = np.stack([v1, v2], axis=1)
    else:
        H = np.stack([v2, v1], axis=1)
    H /= np.linalg.norm(H, axis=0, keepdims=True).clip(min=1e-8)
    return H


def macenko_fit(img):
    """Obtiene el perfil de tinción y la escala de concentración de una imagen."""
    stain = estimate_stain_matrix(img)
    concentrations = np.linalg.lstsq(stain, od(img).reshape(-1, 3).T, rcond=None)[0]
    maxima = np.percentile(concentrations, 99, axis=1)
    return stain, np.maximum(maxima, 1e-6)


def macenko_transform(img, target_stain, target_max):
    """Traslada las concentraciones de la imagen al perfil de referencia.

    Lo que cambia es la apariencia cromática. No deberíamos suponer que
    las estructuras del tejido quedan mejor segmentadas hasta revisarlas.
    """
    src_stain, src_max = macenko_fit(img)
    concentrations = np.linalg.lstsq(src_stain, od(img).reshape(-1, 3).T, rcond=None)[0]
    concentrations *= (target_max / src_max)[:, None]
    output = 255 * np.exp(-(target_stain @ concentrations).T)
    return np.uint8(np.clip(output.reshape(img.shape), 0, 255))


def reinhard_stats(img):
    """Calcula medias y desviaciones de los tres canales Lab de OpenCV.

    Se trata de una adaptación práctica en Lab, no de una reproducción
    literal del espacio de color usado en el método Reinhard original.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(0), np.maximum(lab.reshape(-1, 3).std(0), 1e-5)


def reinhard_transform(img, target_mean, target_std):
    """Ajusta las estadísticas de color para que se acerquen a la referencia."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = lab.reshape(-1, 3).mean(0)
    std = np.maximum(lab.reshape(-1, 3).std(0), 1e-5)
    transformed = (lab - mean) / std * target_std + target_mean
    return cv2.cvtColor(np.uint8(np.clip(transformed, 0, 255)), cv2.COLOR_LAB2RGB)


def vahadane_normalizer(reference):
    """Implementación auténtica, opcional, delegada a la biblioteca staintools.

    staintools suele requerir SPAMS, difícil de instalar en algunos equipos.
    Nunca se sustituye silenciosamente por NMF no equivalente.
    """
    try:
        import staintools
    except (ImportError, OSError) as exc:
        raise RuntimeError('Vahadane requiere staintools y dependencias (incluida SPAMS). '
                           'Instala extras si son compatibles con tu equipo.') from exc
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(reference)
    return normalizer


# -------------------------------------------------------------------------
# CONVERSIÓN DE COLOR Y CREACIÓN DE MÁSCARAS
# -------------------------------------------------------------------------

def otsu_mask(channel, invert=False):
    """Calcula un umbral con Otsu, sin fijarlo manualmente para cada imagen.

    invert=True invierte la selección. El umbral refleja una separación por
    intensidad, no una clasificación de estructuras histológicas.
    """
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(channel, 0, 255, flag | cv2.THRESH_OTSU)
    return mask


def morphological_clean(mask, kernel=3, min_area=35):
    """Suaviza una máscara binaria y elimina componentes muy pequeños.

    La apertura reduce puntos sueltos; el cierre conecta o rellena huecos
    pequeños. Después dejamos solo regiones de al menos min_area píxeles.
    Un valor razonable para 40X no necesariamente lo es para 100X.
    """
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, element)
    count, lab, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    keep = np.zeros_like(cleaned)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def analyze(img, kernel, min_area):
    """Convierte una imagen a HSV/Lab y crea las dos máscaras exploratorias.

    HSV-S privilegia zonas de mayor saturación. Lab-b invertido plantea una
    segunda forma de separar colores. Guardamos ambas para compararlas, no
    para dar por hecho que alguna identifica núcleos correctamente.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Máscara por saturación: tejido coloreado frente a fondo claro.
    mask_hsv = otsu_mask(hsv[:, :, 1])
    # Lab 'b' es solo un candidato exploratorio, NO un detector nuclear validado.
    mask_lab = otsu_mask(lab[:, :, 2], invert=True)
    clean_hsv = morphological_clean(mask_hsv, kernel, min_area)
    clean_lab = morphological_clean(mask_lab, kernel, min_area)
    return {'hsv_h': hsv[:, :, 0], 'hsv_s': hsv[:, :, 1], 'hsv_v': hsv[:, :, 2],
            'lab_l': lab[:, :, 0], 'lab_a': lab[:, :, 1], 'lab_b': lab[:, :, 2],
            'mascara_hsv': mask_hsv, 'mascara_lab': mask_lab,
            'limpia_hsv': clean_hsv, 'limpia_lab': clean_lab}


def save_gray(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', arr)
    if not ok:
        raise IOError(path)
    data.tofile(str(path))


def save_montage(path, rgb, normalized, maps):
    """Crea un panel visual: original, normalizada, canales y máscaras.

    Es una revisión rápida para detectar imágenes demasiado claras,
    saturaciones extrañas o máscaras que han eliminado tejido útil.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells = [('RGB original', rgb), ('Normalizada', normalized),
             ('HSV: saturación', maps['hsv_s']), ('Lab: canal b', maps['lab_b']),
             ('Máscara HSV', maps['mascara_hsv']), ('Máscara Lab', maps['mascara_lab']),
             ('HSV limpiada', maps['limpia_hsv']), ('Lab limpiada', maps['limpia_lab'])]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for axes, (title, data) in zip(ax.flat, cells):
        axes.imshow(data, cmap='gray' if data.ndim == 2 else None)
        axes.set_title(title, fontsize=10)
        axes.axis('off')
    fig.suptitle('BreaKHis 40X/100X: proceso exploratorio', fontsize=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145, bbox_inches='tight')
    plt.close(fig)


def assign_patient_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Separa pacientes, NO fotografías. Las imágenes hermanas quedan juntas.

    El reparto aproximado es 70 % entrenamiento, 15 % validación, 15 % prueba.
    Si no puede identificarse a un paciente, se excluye del procesamiento.
    """
    df = df.copy()
    df['particion'] = 'excluida'
    usable = df[(df['estado'] == 'ok') &
                (df['clase'].isin(['benigna', 'maligna'])) &
                (df['paciente'].astype(str).str.len() > 0)]
    patients = usable[['paciente', 'clase']].drop_duplicates()
    conflicts = patients.groupby('paciente')['clase'].nunique()
    if (conflicts > 1).any():
        raise RuntimeError('Hay identificadores de paciente con dos clases. Revisa el inventario.')
    for cls in ['benigna', 'maligna']:
        ids = sorted(patients.loc[patients['clase'] == cls, 'paciente'].tolist())
        if len(ids) < 5:
            raise RuntimeError(f'Solo hay {len(ids)} pacientes de clase {cls}. '
                               'No puede hacerse una partición por paciente confiable.')
        train_ids, temp_ids = train_test_split(ids, test_size=0.30, random_state=seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)
        for name, subset in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            df.loc[(df['clase'] == cls) & df['paciente'].isin(subset) &
                   (df['estado'] == 'ok'), 'particion'] = name
    return df


def pick_reference(valid: pd.DataFrame, explicit: Path | None):
    """La referencia SIEMPRE procede de un paciente del entrenamiento."""
    train_paths = {str(Path(x).resolve()) for x in valid.loc[valid['particion'] == 'train', 'ruta']}
    if explicit is not None:
        chosen = str(explicit.resolve())
        if chosen not in train_paths:
            raise ValueError('La imagen --reference NO pertenece a train. '
                             'Usa una imagen del entrenamiento para evitar fuga de información.')
        return Path(chosen)
    # Selección reproducible y sencilla; se documenta para que pueda cambiarse.
    candidate = valid[(valid['particion'] == 'train') & (valid['magnificacion'] == '100X')]
    if candidate.empty:
        candidate = valid[valid['particion'] == 'train']
    return Path(candidate.sort_values('ruta').iloc[0]['ruta'])


# -------------------------------------------------------------------------
# EJECUCIÓN DEL PROGRAMA
# -------------------------------------------------------------------------

def main():
    """Organiza el trabajo: inventario -> particiones -> métodos -> archivos."""
    parser = argparse.ArgumentParser(description='BreaKHis 40X/100X: semanas 1 a 5')
    parser.add_argument('--dataset', type=Path, required=True, help='Carpeta raíz con las imágenes originales')
    parser.add_argument('--output', type=Path, default=Path('resultados'))
    parser.add_argument('--reference', type=Path, help='Imagen de referencia de un paciente de train')
    parser.add_argument('--method', choices=['none','reinhard','macenko','vahadane','all'], default='all')
    parser.add_argument('--limit', type=int, default=10, help='N imágenes originales; 0 procesa todas')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel', type=int, default=3, help='Tamaño impar del elemento morfológico')
    parser.add_argument('--min-area', type=int, default=35, help='Píxeles mínimos por región')
    parser.add_argument('--panel-limit', type=int, default=10,
                        help='Paneles visuales por método; 0 no genera paneles')
    args = parser.parse_args()
    if args.limit < 0 or args.panel_limit < 0 or args.min_area < 1 or args.kernel < 1:
        parser.error('El límite, el número de paneles y los tamaños no pueden ser negativos.')
    if not args.dataset.is_dir():
        parser.error(f'No encuentro la base de datos: {args.dataset.resolve()}')
    files = sorted((p for p in args.dataset.rglob('*') if p.is_file() and
                    p.suffix.lower() in EXTS), key=lambda p: str(p))
    rows = [m for p in files if (m := metadata(p)) is not None]
    if not rows:
        parser.error('No se detectaron imágenes 40X o 100X. Revisa el nombre de los archivos.')
    args.output.mkdir(parents=True, exist_ok=True)

    # Inventariar TODO el conjunto, incluso cuando --limit sea 10.
    inventory = []
    for row in rows:
        try:
            rgb = read_rgb(Path(row['ruta']))
            h, w = rgb.shape[:2]
            inventory.append({**row, 'ancho': w, 'alto': h, 'estado': 'ok'})
        except Exception as exc:
            inventory.append({**row, 'ancho': None, 'alto': None, 'estado': f'error: {exc}'})
    df = pd.DataFrame(inventory)
    # Nombres repetidos dentro de la misma magnificación causarían sobrescritura.
    collisions = df.duplicated(['magnificacion', 'archivo'], keep=False)
    df.loc[collisions, 'estado'] = 'excluida: nombre de archivo duplicado'
    # La separación por paciente se hace ANTES de elegir la imagen patrón;
    # así evitamos seleccionar por accidente una referencia del conjunto de prueba.
    df = assign_patient_splits(df, args.seed)
    df.to_csv(args.output/'inventario_40x_100x.csv', index=False, encoding='utf-8-sig')
    pd.crosstab(df['magnificacion'], df['clase']).to_csv(
        args.output/'conteos.csv', encoding='utf-8-sig')
    df.groupby(['magnificacion', 'clase', 'particion'], dropna=False).size().rename('n').to_csv(
        args.output/'conteos_particiones.csv', encoding='utf-8-sig')

    valid = df[df['particion'].isin(['train', 'val', 'test'])].copy()
    if valid.empty:
        raise RuntimeError('No hay imágenes con clase y paciente válidos.')
    # Elegimos una sola imagen de referencia y dejamos anotada su ruta.
    # En una validación cruzada estricta, la referencia también debe ajustarse
    # dentro de cada fold; estos resultados de preprocesamiento son exploratorios.
    refpath = pick_reference(valid, args.reference)
    reference = read_rgb(refpath)
    methods = ['none', 'reinhard', 'macenko', 'vahadane'] if args.method == 'all' else [args.method]
    target_mean, target_std = reinhard_stats(reference)
    mac_target = None
    vah_target = None
    omissions = []
    if 'macenko' in methods:
        try:
            mac_target = macenko_fit(reference)
        except Exception as exc:
            omissions.append(f'Macenko no inicializado: {exc}')
    if 'vahadane' in methods:
        try:
            vah_target = vahadane_normalizer(reference)
        except Exception as exc:
            omissions.append(f'Vahadane no inicializado: {exc}')
    for omission in omissions:
        print('AVISO:', omission)

    # El límite cuenta fotografías originales, no normalizaciones ni máscaras.
    # Selección determinista intercalando 40X/100X y benignas/malignas para las pruebas.
    parts = [grp.sort_values('ruta') for _, grp in valid.groupby(['magnificacion', 'clase'])]
    ordered = []
    for ix in range(max(map(len, parts))):
        for part in parts:
            if ix < len(part):
                ordered.append(part.iloc[ix])
    # Primero elegimos fotografías distintas; después se aplican todos los
    # métodos solicitados a cada una. Por eso --limit 10 son 10 originales,
    # no 10 máscaras ni 10 versiones de tinción.
    selected = ordered if args.limit == 0 else ordered[:args.limit]
    records, errors, saved_panels = [], [], {method: 0 for method in methods}
    for number, row in enumerate(selected, start=1):
        try:
            original = read_rgb(Path(row['ruta']))
        except Exception as exc:
            errors.append({'archivo': row['archivo'], 'metodo': 'lectura', 'error': str(exc)})
            continue
        for method in methods:
            if method == 'macenko' and mac_target is None:
                continue
            if method == 'vahadane' and vah_target is None:
                continue
            try:
                if method == 'none':
                    normalized = original
                elif method == 'reinhard':
                    normalized = reinhard_transform(original, target_mean, target_std)
                elif method == 'macenko':
                    normalized = macenko_transform(original, *mac_target)
                else:
                    normalized = vah_target.transform(original)
                maps = analyze(normalized, args.kernel, args.min_area)
                # Guardar cada condición en su propia carpeta permite volver
                # a encontrar la imagen y su máscara en la semana 6.
                destination = args.output / method / row['magnificacion'] / Path(row['archivo']).stem
                write_rgb(destination/'normalizada.png', normalized)
                for key, value in maps.items():
                    save_gray(destination/f'{key}.png', value)
                if saved_panels[method] < args.panel_limit:
                    save_montage(destination/'comparacion.png', original, normalized, maps)
                    saved_panels[method] += 1
                records.append({'archivo': row['archivo'], 'magnificacion': row['magnificacion'],
                                'clase': row['clase'], 'paciente': row['paciente'],
                                'particion': row['particion'], 'metodo': method,
                                'porcentaje_mascara_hsv': float(np.mean(maps['limpia_hsv'] > 0)*100),
                                'porcentaje_mascara_lab': float(np.mean(maps['limpia_lab'] > 0)*100)})
                print(f'[{number}/{len(selected)}] {method}: {row["archivo"]}')
            except Exception as exc:
                errors.append({'archivo': row['archivo'], 'metodo': method, 'error': str(exc)})
                print(f'ERROR [{method}] {row["archivo"]}: {exc}')
    pd.DataFrame(records, columns=['archivo', 'magnificacion', 'clase', 'paciente', 'particion',
                                  'metodo', 'porcentaje_mascara_hsv', 'porcentaje_mascara_lab']).to_csv(
        args.output/'resumen_mascaras.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(errors, columns=['archivo','metodo','error']).to_csv(
        args.output/'errores_preprocesamiento.csv', index=False, encoding='utf-8-sig')
    config = {'dataset': str(args.dataset.resolve()), 'reference': str(refpath.resolve()),
              'reference_from_partition': 'train', 'seed': args.seed,
              'images_selected': len(selected), 'records_written': len(records),
              'methods_requested': methods, 'omissions': omissions,
              'kernel': args.kernel, 'min_area': args.min_area}
    (args.output/'configuracion.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nTerminó: {len(selected)} imágenes distintas, {len(records)} combinaciones guardadas.')
    print(f'Inventario completo: {len(df)} imágenes. Resultados en: {args.output.resolve()}')
    print('Las máscaras son aproximaciones por color, no segmentaciones nucleares validadas.')


if __name__ == '__main__':
    main()
