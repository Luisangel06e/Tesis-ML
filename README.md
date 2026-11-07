# Estancia académica — Análisis histopatológico de BreaKHis (40X y 100X)

Este repositorio reúne el código, la documentación y los resultados de las **semanas 1 a 7** de la estancia académica. El objetivo del trabajo es preparar imágenes histopatológicas de mama, extraer características de color, textura y geometría, y comparar modelos supervisados para distinguir entre imágenes etiquetadas como **benignas** y **malignas**.

Se trabaja únicamente con las magnificaciones **40X y 100X** de BreaKHis. Las magnificaciones 200X y 400X no se incluyen por una **delimitación metodológica**, no porque carezcan de utilidad diagnóstica.

> **Alcance:** los programas son herramientas de investigación; no constituyen un sistema de diagnóstico clínico. Las máscaras obtenidas por color son exploratorias y no equivalen a una segmentación nuclear validada.

## Organización del repositorio

| Archivo o carpeta | Contenido |
| --- | --- |
| `preprocesamiento.py` | Lectura, selección y preprocesamiento de imágenes (semanas 1–5). |
| `extraccion_caracteristicas.py` | Extracción de descriptores de color, GLCM, LBP y geometría (semana 6). |
| `Modelo Supervisados.py` | Evaluación de SVM, Random Forest y XGBoost; métricas y gráficas (semana 7). |
| `requirements.txt` | Bibliotecas necesarias para ejecutar los scripts. |
| `Documentación/` | Informes y documentos de la estancia. |
| `resultados_semana_6/` | Archivos de características y resúmenes generados en la semana 6. |
| `resultados_semana7/` | Métricas, predicciones y gráficas de la semana 7. |

**Nota sobre nombres:** las primeras versiones del proyecto utilizaban `analisis_breakhis.py`. En el estado actual del repositorio, ese archivo figura como eliminado y aparece `preprocesamiento.py`. Los ejemplos siguientes utilizan el nombre actual. Verifica las opciones con `python preprocesamiento.py --help` si cambiaste el contenido del programa al renombrarlo.

## Base de datos y distribución de referencia

La siguiente distribución corresponde al subconjunto de magnificaciones 40X y 100X reportado en la literatura; debe **contrastarse con el inventario real** de la copia utilizada.

| Clase | 40X | 100X | Total |
| --- | ---: | ---: | ---: |
| Benigna | 625 | 644 | 1,269 |
| Maligna | 1,370 | 1,437 | 2,807 |
| **Total** | **1,995** | **2,081** | **4,076** |

Si el inventario o los archivos de características arrojan una cantidad diferente, hay que investigar la diferencia antes de reportar los resultados como completos. Los conteos generados por una fase de procesamiento no demuestran, por sí solos, que falten imágenes en la base original.

## Metodología

### Semanas 1–5. Selección, normalización y segmentación exploratoria

**Selección y análisis exploratorio.** El programa revisa nombres y carpetas de BreaKHis para identificar la magnificación, la clase y, cuando es posible, el paciente. Registra las imágenes localizadas y sus dimensiones en `inventario_40x_100x.csv`, y resume su distribución por clases y magnificaciones.

**Revisión y selección de métodos.** Se comparan condiciones de normalización para evaluar su efecto sobre la apariencia de la tinción H&E. No se supone que un método sea mejor solo por producir imágenes visualmente uniformes: debe comprobarse su impacto en las características y, posteriormente, en los modelos.

- **Sin normalización (`none`):** condición de referencia para comparar con las imágenes originales.
- **Reinhard:** transferencia estadística de color mediante canales Lab codificados por OpenCV. Es una adaptación práctica; no reproduce exactamente el espacio de color del método original [2].
- **Macenko:** transformación a densidad óptica, estimación de vectores de tinción y ajuste de concentraciones respecto de una referencia [3].
- **Vahadane:** método opcional de separación de tinciones. Solo se ejecuta si `staintools` y sus dependencias están disponibles; si no, se omite y no se reporta como realizado [4].

**Conversión de color.** Se estudian los espacios **HSV** (tono, saturación y valor) y **Lab** (luminosidad y canales cromáticos). Los valores Lab de 8 bits producidos por OpenCV no deben confundirse directamente con las coordenadas físicas de CIELAB.

**Umbralización y limpieza.** Se utiliza Otsu sobre canales de color para generar máscaras exploratorias. Después se aplican apertura, cierre y eliminación de componentes pequeños. Las máscaras pueden incluir tejido, fondo o artefactos y deben revisarse visualmente. El porcentaje de píxeles seleccionados **no mide la exactitud de la segmentación**.

| Condición | Normalización | Máscaras exploratorias |
| --- | --- | --- |
| N0 | Ninguna | HSV y Lab |
| N1 | Reinhard | HSV y Lab |
| N2 | Macenko | HSV y Lab |
| N3 | Vahadane (si está disponible) | HSV y Lab |

### Semana 6. Extracción de características histomorfológicas

El segundo programa lee las imágenes procesadas y sus máscaras para generar descriptores numéricos:

- **Color:** estadísticas de canales RGB, HSV y Lab.
- **Textura:** características basadas en la matriz de coocurrencia de niveles de gris (GLCM) y patrones binarios locales (LBP).
- **Geometría:** área, perímetro, circularidad, solidez, relación de aspecto y excentricidad de regiones conectadas.

Las medidas geométricas están expresadas en **píxeles** y no se deben interpretar directamente como medidas físicas equivalentes entre 40X y 100X. Una región conectada de la máscara no representa necesariamente un núcleo celular. La corrección del cálculo geométrico busca evitar valores imposibles al mezclar definiciones distintas de área y perímetro.

Se generan `caracteristicas_por_imagen.csv`, `caracteristicas_por_region.csv`, `resumen_grupos.csv` y `errores.csv`.

### Semana 7. Modelado supervisado y evaluación

El tercer programa compara **SVM**, **Random Forest** y **XGBoost**. Examina 40X, 100X y el conjunto combinado, y calcula métricas como exactitud, precisión, sensibilidad, especificidad, F1 y AUC-ROC. También genera matrices de confusión, curvas ROC, gráficas comparativas e importancia de variables cuando el modelo la admite.

La evaluación debe separar a los pacientes entre entrenamiento y validación para impedir que fotografías del mismo paciente aparezcan en ambos grupos. Si el CSV conserva una columna de partición, la configuración predeterminada del script reserva el conjunto `test` y realiza validación cruzada sobre `train`. Esto **no sustituye** una evaluación independiente final ni la validación externa.

Los archivos principales de salida son `metricas_por_fold.csv`, `resumen_modelos.csv`, `predicciones_fuera_de_fold.csv`, `metadatos_ejecucion.json` y la carpeta `graficas/`.

## Instalación y ejecución en Windows

Abre **CMD** en la carpeta del proyecto:

```cmd
cd /d "C:\Users\Luis Diaz\OneDrive\Documentos\Escritorio\estancia\Estancia"
python -m pip install -r requirements.txt
```

Es recomendable crear un entorno virtual para que las dependencias del proyecto no afecten otras instalaciones de Python:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 1. Preprocesamiento

Prueba primero con diez imágenes originales:

```cmd
python preprocesamiento.py --dataset "C:\RUTA\BreaKHis" --output resultados_preprocesamiento --limit 10 --method all
```

Para procesar todas las imágenes reconocidas de 40X y 100X, cambia a `--limit 0`. El parámetro `--limit` cuenta **fotografías originales**, no las combinaciones de método y máscara. Para una comparación controlada, la imagen de referencia de tinción debe pertenecer al conjunto de entrenamiento.

### 2. Extracción de características

```cmd
python extraccion_caracteristicas.py --results resultados_preprocesamiento --output resultados_semana_6 --limit 0
```

La extracción requiere la estructura de archivos e inventario producidos por el programa de preprocesamiento. Si se utiliza una carpeta anterior, verifica que sus nombres y organización sean compatibles.

### 3. Modelos supervisados con gráficas

```cmd
python "Modelo Supervisados.py" --csv "resultados_semana_6\caracteristicas_por_imagen.csv" --output resultados_semana7 --folds 5
```

El programa puede producir carpetas de gráficas separadas para **40X**, **100X** y **40X+100X**. Si alguna magnificación no tiene pacientes suficientes para la validación por grupos, revisa los datos y considera menos folds, en vez de mezclar imágenes del mismo paciente.

Para comprobar las opciones disponibles en cada versión:

```cmd
python preprocesamiento.py --help
python extraccion_caracteristicas.py --help
python "Modelo Supervisados.py" --help
```

## Reproducibilidad y precauciones

1. Mantener las imágenes originales intactas y guardar los resultados en directorios separados.
2. Registrar la semilla aleatoria, la versión del código, las particiones por paciente y la imagen de referencia de tinción.
3. No mezclar métricas ni gráficas procedentes de diferentes ejecuciones sin indicar la configuración utilizada.
4. Excluir las muestras sin identificador de paciente de experimentos que requieran validación agrupada, salvo que se resuelva su identidad.
5. Evitar interpretar una máscara de color como segmentación histológica validada; para medir su calidad se necesitan anotaciones de referencia.
6. Reconocer que seleccionar una referencia de normalización de `train` no constituye una validación cruzada totalmente anidada: para esa evaluación habría que ajustar las decisiones de normalización dentro de cada fold.
7. Antes de publicar resultados, revisar el tamaño y la procedencia de los CSV, y confirmar que no contienen información confidencial o identificadores personales no autorizados.

## Referencias metodológicas

1. *Lightweight Separable Convolution Network for Breast Cancer Histopathological Identification*. (2023). **Diagnostics**, 13(2), 299. Distribución de BreaKHis: <https://www.mdpi.com/2075-4418/13/2/299>
2. Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications, 21*(5), 34–41.
3. Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE ISBI*. <https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf>
4. Vahadane, A., et al. (2016). Structure-preserving color normalization and sparse stain separation for histological images. *IEEE Transactions on Medical Imaging, 35*(8), 1962–1971. <https://doi.org/10.1109/TMI.2016.2529665>
5. OpenCV. Documentación de conversiones de color, umbralización y transformaciones morfológicas. <https://docs.opencv.org/>
