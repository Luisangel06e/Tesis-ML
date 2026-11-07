#!/usr/bin/env python3
"""Semana 6 | Extracción de características de BreaKHis 40X y 100X.

Este programa recoge las imágenes y máscaras de las semanas 1 a 5 y las
convierte en datos numéricos. Calcula valores de color, textura (GLCM y LBP)
y geometría de las regiones detectadas por la máscara.

Genera un CSV por imagen, otro por región, un resumen y un registro de errores.
No realiza diagnósticos ni identifica núcleos de manera validada: una región
blanca puede ser tejido, un grupo de células, fondo o un artefacto.

Ejemplo:
    python extraccion_caracteristicas.py --results resultados --output resultados_punto_6 --limit 10

Usa --limit 0 para analizar todas las imágenes que se procesaron previamente.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern

# Estos nombres coinciden con los archivos que guarda analisis_breakhis.py.
# Por eso no hace falta volver a ejecutar la normalización en esta semana.
MASKS = {'hsv': 'limpia_hsv.png', 'lab': 'limpia_lab.png'}
METHODS = ('none', 'reinhard', 'macenko', 'vahadane')


def read_img(path: Path, gray=False):
    """Lee fotografías en color o máscaras en gris incluso con rutas Unicode."""
    b = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(b, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Archivo de imagen ilegible: {path}')
    return img


def safe_float(x):
    """Devuelve NaN si un cálculo no es válido, en lugar de inventar un cero."""
    return float(x) if np.isfinite(x) else np.nan


# -------------------------------------------------------------------------
# TEXTURA: MATRIZ DE COOCURRENCIA (GLCM)
# -------------------------------------------------------------------------

def masked_glcm(gray: np.ndarray, mask: np.ndarray, levels=16):
    """GLCM de parejas donde ambos píxeles están dentro de la máscara.

    Las distancias son en píxeles; no son equivalentes físicamente entre 40X/100X.
    """
    # Reducimos de 256 tonos de gris a 16. Con menos niveles, el cálculo de
    # coocurrencias usa menos memoria y resulta más manejable.
    q = np.minimum(gray.astype(np.int32) * levels // 256, levels - 1)
    valid = mask > 0
    # Repetimos el cálculo en cuatro direcciones y dos distancias (1 y 2 px).
    # No conviene comparar estas distancias en términos físicos entre 40X y
    # 100X sin calibrar primero la resolución de los microscopios.
    offsets = [(0,1), (1,1), (1,0), (1,-1),
               (0,2), (2,2), (2,0), (2,-2)]
    cs, hs, es, rs = [], [], [], []
    idx = np.arange(levels, dtype=np.float64)
    ii, jj = np.meshgrid(idx, idx, indexing='ij')
    for dy, dx in offsets:
        h, w = gray.shape
        ya = slice(max(0,dy), min(h,h+dy))
        yb = slice(max(0,-dy), min(h,h-dy))
        xa = slice(max(0,dx), min(w,w+dx))
        xb = slice(max(0,-dx), min(w,w-dx))
        # Solo contamos pares de píxeles si AMBOS caen en la máscara.
        # De otro modo, el fondo claro distorsionaría la textura del tejido.
        pair = valid[ya,xa] & valid[yb,xb]
        if not np.any(pair):
            continue
        codes = q[ya,xa][pair] * levels + q[yb,xb][pair]
        glcm = np.bincount(codes, minlength=levels**2).reshape(levels, levels).astype(np.float64)
        glcm += glcm.T
        glcm /= glcm.sum()
        di = ii-jj
        cs.append(np.sum(glcm * di**2))
        hs.append(np.sum(glcm / (1 + di**2)))
        es.append(np.sqrt(np.sum(glcm**2)))
        mx, my = np.sum(ii*glcm), np.sum(jj*glcm)
        sx = np.sqrt(np.sum((ii-mx)**2*glcm))
        sy = np.sqrt(np.sum((jj-my)**2*glcm))
        rs.append(np.sum((ii-mx)*(jj-my)*glcm)/(sx*sy) if sx*sy > 1e-12 else np.nan)
    def mean_or_nan(items):
        a = np.asarray(items, dtype=float)
        return safe_float(np.nanmean(a)) if a.size and np.any(np.isfinite(a)) else np.nan
    return {'glcm_contraste': mean_or_nan(cs),
            'glcm_homogeneidad': mean_or_nan(hs),
            'glcm_energia': mean_or_nan(es),
            'glcm_correlacion': mean_or_nan(rs)}


# -------------------------------------------------------------------------
# GEOMETRÍA: COMPONENTES CONECTADOS DE LAS MÁSCARAS
# -------------------------------------------------------------------------

def region_features(mask: np.ndarray, min_region_area: int):
    """Describe cada región blanca suficientemente grande de la máscara.

    Los números de área y perímetro están en píxeles. Son descriptores
    geométricos de componentes, NO medidas confirmadas de núcleos celulares.
    """
    mask = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_region_area:
            continue
        # Trabajamos solo dentro del rectángulo de la región. Así evitamos crear
        # una máscara del tamaño completo de la imagen para cada componente.
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        ww = int(stats[lab, cv2.CC_STAT_WIDTH])
        hh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        component = np.uint8(labels[y0:y0+hh, x0:x0+ww] == lab) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        # Mantener la MISMA definición geométrica del área en solidez y circularidad.
        # CC_STAT_AREA cuenta píxeles, mientras cv2.contourArea estima área del contorno.
        # Su mezcla producía índices mayores que 1.
        contour_area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        x,y,w,h = cv2.boundingRect(contour)
        # Los momentos espaciales permiten estimar qué tan alargada es la
        # región; una excentricidad alta sugiere una forma más elongada.
        moments = cv2.moments(component, binaryImage=True)
        if moments['m00'] > 0:
            mu20 = moments['mu20']/moments['m00']
            mu02 = moments['mu02']/moments['m00']
            mu11 = moments['mu11']/moments['m00']
            vals = np.linalg.eigvalsh([[mu20,mu11],[mu11,mu02]])
            eccentricity = float(np.sqrt(max(0.0, 1 - max(vals[0],0) / vals[1]))) if vals[1] > 0 else np.nan
        else:
            eccentricity = np.nan
        rows.append({
            'region_id': lab, 'area_px': area, 'perimetro_px': perimeter,
            'circularidad': safe_float(4 * np.pi * contour_area / perimeter**2) if perimeter > 0 and contour_area > 0 else np.nan,
            'solidez': safe_float(contour_area / hull_area) if hull_area > 0 and contour_area > 0 else np.nan,
            'relacion_aspecto': float(w/h) if h else np.nan,
            'excentricidad': eccentricity,
        })
    return rows


# -------------------------------------------------------------------------
# CARACTERÍSTICAS QUE RESUMEN CADA IMAGEN
# -------------------------------------------------------------------------

def image_features(bgr, mask, min_region_area=35):
    """Une color, textura y geometría en una sola fila de la tabla final."""
    mask = mask > 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    regions = region_features(np.uint8(mask)*255, min_region_area)
    out = {'ancho': bgr.shape[1], 'alto': bgr.shape[0],
           'cobertura_mascara_pct': float(mask.mean() * 100),
           'n_regiones': len(regions),
           'area_media_region_px': safe_float(np.mean([x['area_px'] for x in regions])) if regions else np.nan,
           'circularidad_media': safe_float(np.mean([x['circularidad'] for x in regions])) if regions else np.nan,
           'solidez_media': safe_float(np.mean([x['solidez'] for x in regions])) if regions else np.nan}
    # Para cada canal calculamos media y dispersión SOLO en los píxeles
    # seleccionados. Esto describe el color de la región, no de todo el fondo.
    for family, img, names in [('rgb',rgb,'rgb'), ('hsv',hsv,'hsv'), ('lab',lab,'lab')]:
        for ch,name in enumerate(names):
            values = img[:,:,ch][mask]
            out[f'{family}_{name}_media'] = float(values.mean()) if values.size else np.nan
            out[f'{family}_{name}_desv'] = float(values.std()) if values.size else np.nan
    # Las texturas necesitan suficientes píxeles para producir resultados
    # interpretables. Si la máscara queda casi vacía, guardamos NaN.
    if mask.sum() >= 20:
        out.update(masked_glcm(gray, mask))
        # Uniform LBP, radio=1, P=8, 10 categorías (0..9).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist = np.bincount(np.asarray(lbp[mask], dtype=np.int32), minlength=10)[:10].astype(float)
        hist /= max(hist.sum(), 1)
        out['lbp_entropia'] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        for k, value in enumerate(hist):
            out[f'lbp_bin_{k}'] = float(value)
    else:
        out.update({k: np.nan for k in ('glcm_contraste','glcm_homogeneidad','glcm_energia','glcm_correlacion','lbp_entropia')})
        out.update({f'lbp_bin_{k}': np.nan for k in range(10)})
    return out, regions


# -------------------------------------------------------------------------
# LECTURA DE RESULTADOS Y EXPORTACIÓN A CSV
# -------------------------------------------------------------------------

def main():
    """Recorre las condiciones disponibles y genera las tablas de la semana 6."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True, help='Carpeta resultados de etapas 1-5')
    p.add_argument('--output', type=Path, default=Path('resultados_punto_6'))
    p.add_argument('--method', choices=(*METHODS,'all'), default='all')
    p.add_argument('--mask', choices=('hsv','lab','both'), default='both')
    p.add_argument('--limit', type=int, default=10, help='0=todas; positivo=N fotografías ORIGINALES')
    p.add_argument('--min-region-area', type=int, default=35)
    args = p.parse_args()
    if args.limit < 0 or args.min_region_area < 1:
        p.error('Revisa --limit y --min-region-area')
    inventory_path = args.results / 'inventario_40x_100x.csv'
    if not inventory_path.exists():
        p.error(f'No existe {inventory_path}. Ejecuta antes etapas 1-5.')
    df = pd.read_csv(inventory_path, dtype=str).fillna('')
    required = {'archivo','magnificacion','ruta','clase','paciente','subtipo','estado'}
    if not required.issubset(set(df.columns)):
        p.error(f'Inventario incompleto, faltan: {sorted(required - set(df.columns))}')
    df = df[df['magnificacion'].isin(['40X','100X']) & (df['estado']=='ok')].copy()
    # En los resultados nuevos ya existe 'particion'. Para los anteriores dejamos
    # la columna vacía en lugar de atribuirles una partición que desconocemos.
    if 'particion' not in df.columns:
        df['particion'] = ''
    # Una misma fotografía no debe aparecer repetida en el inventario.
    # Las versiones normalizadas se buscan después, por separado.
    if not df['ruta'].is_unique:
        p.error('El inventario contiene rutas duplicadas; es necesario revisarlo.')
    # Elige N imágenes distintas primero y después procesa TODAS sus condiciones.
    # Esta es la diferencia frente a la versión anterior, cuyo límite contaba filas
    # imagen/método/máscara, algo que resultaba confuso.
    df = df.sort_values(['magnificacion','clase','archivo']).reset_index(drop=True)
    if args.limit:
        clases = [g for _, g in df.groupby(['magnificacion', 'clase'])]
        selected_rows = []
        for j in range(max(map(len, clases))):
            for g in clases:
                if j < len(g):
                    selected_rows.append(g.iloc[j])
                    if len(selected_rows) >= args.limit:
                        break
            if len(selected_rows) >= args.limit:
                break
        df = pd.DataFrame(selected_rows)

    methods = METHODS if args.method == 'all' else (args.method,)
    masks = MASKS if args.mask == 'both' else {args.mask:MASKS[args.mask]}
    # Solo añadimos combinaciones que EXISTEN en disco. Si Vahadane no se
    # pudo instalar en las semanas 1-5, aquí no fabricamos sus resultados.
    files = []
    for method in methods:
        for _, row in df.iterrows():
            folder = args.results / method / row['magnificacion'] / Path(row['archivo']).stem
            norm = folder / 'normalizada.png'
            if not norm.exists():
                continue
            for mask_name, filename in masks.items():
                mask_path = folder / filename
                if mask_path.exists():
                    files.append((method,row.to_dict(),norm,mask_name,mask_path))
    if not files:
        p.error('No se encontraron normalizadas.png y máscaras limpias. Revisa --results y --method.')
    args.output.mkdir(parents=True, exist_ok=True)
    imgs, comps, errors = [], [], []
    for method,row,img_path,mask_name,mask_path in files:
        info = {k:row.get(k,'') for k in ['archivo','magnificacion','clase','subtipo','paciente','particion']}
        info.update({'metodo':method,'tipo_mascara':mask_name})
        try:
            bgr = read_img(img_path)
            mask = read_img(mask_path, gray=True)
            if mask.shape != bgr.shape[:2]:
                raise ValueError('Dimensiones de máscara e imagen distintas')
            features, regions = image_features(bgr, mask, args.min_region_area)
            imgs.append({**info,**features})
            comps.extend({**info,**r} for r in regions)
            print(f"OK {info['magnificacion']} / {method} / {mask_name} / {info['archivo']} ({len(regions)} regiones)")
        except Exception as exc:
            errors.append({**info,'error':str(exc)})
            print(f"ERROR {info['archivo']} / {method} / {mask_name}: {exc}")
    # Una fila por imagen/método/máscara. Esta será la entrada de SVM,
    # Random Forest y XGBoost en la semana 7.
    pd.DataFrame(imgs).to_csv(args.output/'caracteristicas_por_imagen.csv',index=False,encoding='utf-8-sig')
    # En la tabla de regiones puede haber muchas filas por fotografía.
    # No la usamos directamente como si cada región fuera otro paciente.
    pd.DataFrame(comps,columns=[*['archivo','magnificacion','clase','subtipo','paciente','particion','metodo','tipo_mascara'],
                               'region_id','area_px','perimetro_px','circularidad','solidez','relacion_aspecto','excentricidad']).to_csv(
        args.output/'caracteristicas_por_region.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(errors,columns=['archivo','magnificacion','clase','subtipo','paciente','particion','metodo','tipo_mascara','error']).to_csv(
        args.output/'errores.csv',index=False,encoding='utf-8-sig')
    if imgs:
        summary = (pd.DataFrame(imgs).groupby(['magnificacion','clase','metodo','tipo_mascara'],dropna=False)
                   .agg(n_imagenes=('archivo','size'), cobertura_media_pct=('cobertura_mascara_pct','mean'),
                        regiones_media=('n_regiones','mean'), glcm_contraste_medio=('glcm_contraste','mean'),
                        lbp_entropia_media=('lbp_entropia','mean')).reset_index())
        summary.to_csv(args.output/'resumen_grupos.csv',index=False,encoding='utf-8-sig')
    print(f'\nImágenes distintas: {len(set(r["archivo"] for r in imgs))}; combinaciones: {len(imgs)}; errores: {len(errors)}.')
    print(f'Carpeta: {args.output.resolve()}')
    print('AVISO: Las regiones son COMPONENTES DE MÁSCARA, NO NÚCLEOS CONFIRMADOS.')

if __name__ == '__main__':
    main()
