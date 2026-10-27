# Estancia académica | Semanas 1 a 5

## Normalización de tinción y segmentación exploratoria

**Guía metodológica y manual de código | BreaKHis 40X y 100X**

Alcance: etapas 1 a 5 del flujo de preprocesamiento, sin entrenamiento ni clasificación de tumores.

## Resumen

Se propone un procedimiento reproducible para examinar imágenes histopatológicas de mama de BreaKHis, limitado a las magnificaciones 40X y 100X. Se construye un inventario, se aplican condiciones de normalización cromática (sin normalización, Reinhard, Macenko y Vahadane cuando sus dependencias están disponibles), se transforman imágenes a HSV y Lab, se obtienen máscaras binarias y se limpian mediante morfología matemática. Las máscaras son exploratorias: por sí solas no constituyen segmentación nuclear validada.

## Alcance y datos

| Clase | 40X | 100X | Total |
| --- | --- | --- | --- |
| Benigna | 625 | 644 | 1,269 |
| Maligna | 1,370 | 1,437 | 2,807 |
| Total | 1,995 | 2,081 | 4,076 |

Las cantidades se reportan como referencia publicada de BreaKHis [1]. Deben verificarse con el inventario generado sobre la copia descargada. Las magnificaciones 200X y 400X se excluyen deliberadamente.

## Etapa 1. Selección y lectura de BreaKHis 40X/100X

El programa recorre carpetas y busca magnificación en nombres típicos de BreaKHis (p. ej., `SOB_B_A-14-22549AB-40-001.png`) o en carpetas 40X y 100X. Extrae clase, subtipo e identificador del paciente cuando estos datos están presentes en el nombre original. Después comprueba la lectura de la imagen y registra ancho y alto en un CSV.

Salidas principales: `inventario_40x_100x.csv` y `conteos.csv`. Los pacientes sin identificador recuperable quedan señalados con campo vacío; no deben asumirse agrupaciones correctas sin revisión.

## Etapa 2. Normalización de tinción

### 2.1. Reinhard

El script implementa una transferencia estadística de color mediante los canales Lab codificados por OpenCV: ajusta medias y desviaciones estándar de la imagen fuente a las de una referencia. Se trata de una adaptación práctica basada en Lab, no una reproducción exacta del espacio lαβ descrito en el artículo original de Reinhard [2].

### 2.2. Macenko

Se convierten intensidades RGB a densidad óptica, se excluyen píxeles de baja densidad y se estiman vectores de tinción a partir del plano dominante por descomposición espectral. Las concentraciones se reescalan con percentiles de una imagen de referencia y se reconstruye la imagen [3]. Es una implementación didáctica de Macenko; se debe inspeccionar su comportamiento con fondos y muestras particulares.

### 2.3. Vahadane

Vahadane utiliza separación de tinciones con restricciones de no negatividad y dispersión para procurar preservar estructura [4]. Por rigor metodológico, el script llama una implementación de la biblioteca `staintools` únicamente cuando está instalada con sus dependencias. Si no está disponible, el método se omite con una advertencia explícita; **NO** se reporta como ejecutado.

> **Control metodológico:** la imagen de referencia debe elegirse **SOLO** entre imágenes del entrenamiento una vez que exista partición por paciente. La referencia por defecto del script se utiliza únicamente para demostración y activa una advertencia.

## Etapa 3. Conversión RGB a HSV y Lab

En HSV se obtienen tono (H), saturación (S) y valor (V). En Lab se obtienen luminosidad (L), canal verde–rojo (a) y canal azul–amarillo (b). El código guarda los seis canales como imágenes en escala de grises. La codificación de Lab usada por OpenCV es de 8 bits y no debe confundirse directamente con las escalas físicas L\* 0–100, a\* y b\* de CIELAB.

## Etapa 4. Umbralización y creación de máscaras

Se aplica el método de Otsu al canal S de HSV para estudiar separación aproximada entre tejido coloreado y fondo; se aplica Otsu inverso al canal b de Lab como segunda alternativa exploratoria. Los umbrales se determinan por imagen y **NO** han sido calibrados para segmentar núcleos. Se guardan `mascara_hsv.png` y `mascara_lab.png`. Las máscaras deben revisarse comparándolas con la imagen original y, si se pretende medir rendimiento, con anotaciones de referencia.

## Etapa 5. Limpieza mediante operaciones morfológicas

Las máscaras pasan por apertura (erosión seguida de dilatación) para reducir ruido y cierre (dilatación seguida de erosión) para rellenar pequeñas discontinuidades. Posteriormente se eliminan componentes conectados menores que un área mínima. Los parámetros iniciales son kernel elíptico de 3×3 y área mínima de 35 píxeles; deben optimizarse de modo independiente a los datos de prueba y considerar la magnificación de la imagen.

Salidas: `limpia_hsv.png` y `limpia_lab.png`. El porcentaje de píxeles blancos de cada máscara limpia se registra en `resumen_mascaras.csv`, pero **NO** es una métrica de exactitud de segmentación.

## Diseño de comparación sugerido

| Condición | Normalización | Segmentación |
| --- | --- | --- |
| N0 | Ninguna | Otsu en HSV-S y Lab-b |
| N1 | Reinhard-Lab | Otsu en HSV-S y Lab-b |
| N2 | Macenko | Otsu en HSV-S y Lab-b |
| N3 | Vahadane (opcional) | Otsu en HSV-S y Lab-b |

Comparar visualmente al menos imágenes benignas y malignas de 40X y 100X, utilizando la misma referencia dentro de cada corrida. Describir si las máscaras incluyen fondo, se fusionan regiones o pierden detalles. En esta fase no corresponde concluir qué método mejora la detección del cáncer: faltan segmentaciones de referencia y pruebas de clasificación.

## Ejecución del código

1. Instalar Python 3.10 o superior.
2. Abrir terminal en la carpeta entregada.
3. Ejecutar:

```bash
python -m pip install -r requirements.txt

python analisis_breakhis.py --dataset "RUTA_A_BREaKHis" --output resultados --limit 8 --method all
```

Para procesar todas las imágenes se usa `--limit 0`. Para un experimento metodológicamente controlado debe indicarse `--reference "RUTA_A_IMAGEN_DE_TRAIN.png"`. Ejemplo:

```bash
python analisis_breakhis.py --dataset "RUTA_A_BREaKHis" --output resultados --limit 0 --method macenko --reference "RUTA_A_REFERENCIA_TRAIN.png"
```

Opciones adicionales: `--kernel 3` y `--min-area 35`. El script genera un directorio para cada método, magnificación y archivo con imagen normalizada, canales, máscaras y un panel comparativo.

## Limitaciones y criterios de calidad

- No se ha ejecutado el análisis sobre la copia real del usuario: no se reportan resultados empíricos ni tasas de éxito.
- El conjunto tiene múltiples imágenes por paciente. Antes de cualquier modelado se requiere partición agrupada por paciente y control de fuga de información.
- La máscara basada en color puede identificar regiones de tejido, tinción, artefactos o fondo; no equivale a delimitar núcleos o tumores.
- El procesamiento de todas las imágenes y varias condiciones puede generar miles de archivos y consumir espacio y tiempo.
- Vahadane depende de bibliotecas externas que pueden no estar disponibles en Windows o Mac. Registrar en el informe si fue omitido.
- No se calculan sensibilidad, Dice ni IoU de segmentación porque no se suministraron anotaciones de referencia.

## Resultados esperados y entregables

El procedimiento producirá inventario CSV, tabla de conteos por clase/magnificación, carpetas de imágenes normalizadas y canales HSV/Lab, máscaras originales y depuradas, paneles comparativos y un resumen del área seleccionada por las máscaras. Constituye la base para una futura evaluación cuantitativa de segmentación y para la extracción de características histomorfológicas, que se encuentran fuera del alcance actual.

## Referencias

1. Lightweight Separable Convolution Network for Breast Cancer Histopathological Identification. *Diagnostics* (2023). Tabla 1 sobre distribución BreaKHis. <https://www.mdpi.com/2075-4418/13/2/299>
2. Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color Transfer between Images. *IEEE Computer Graphics and Applications*, 21(5), 34–41.
3. Macenko, M. et al. (2009). A Method for Normalizing Histology Slides for Quantitative Analysis. *IEEE ISBI*. <https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf>
4. Vahadane, A. et al. (2016). Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images. *IEEE Transactions on Medical Imaging*, 35(8), 1962–1971. <https://doi.org/10.1109/TMI.2016.2529665>
5. OpenCV documentation. Color conversions; thresholding and morphological transformations. <https://docs.opencv.org/>
