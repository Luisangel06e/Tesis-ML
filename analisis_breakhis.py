
"""BreaKHis 40X/100X: inventario, normalización, HSV/Lab, máscaras y morfología.
"""
from __future__ import annotations
import argparse
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def metadata(path: Path):
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
    # imdecode admite algunas rutas con caracteres Unicode en Windows.
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Imagen ilegible: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f'No se pudo guardar: {path}')
    buf.tofile(str(path))


def od(rgb):
    return -np.log(np.maximum(rgb.astype(np.float64), 1.0) / 255.0)


def estimate_stain_matrix(rgb, alpha=1.0, beta=0.15):
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
    stain = estimate_stain_matrix(img)
    concentrations = np.linalg.lstsq(stain, od(img).reshape(-1, 3).T, rcond=None)[0]
    maxima = np.percentile(concentrations, 99, axis=1)
    return stain, np.maximum(maxima, 1e-6)


def macenko_transform(img, target_stain, target_max):
    src_stain, src_max = macenko_fit(img)
    concentrations = np.linalg.lstsq(src_stain, od(img).reshape(-1, 3).T, rcond=None)[0]
    concentrations *= (target_max / src_max)[:, None]
    output = 255 * np.exp(-(target_stain @ concentrations).T)
    return np.uint8(np.clip(output.reshape(img.shape), 0, 255))


def reinhard_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(0), np.maximum(lab.reshape(-1, 3).std(0), 1e-5)


def reinhard_transform(img, target_mean, target_std):
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


def otsu_mask(channel, invert=False):
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(channel, 0, 255, flag | cv2.THRESH_OTSU)
    return mask


def morphological_clean(mask, kernel=3, min_area=35):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True, help='Carpeta raíz BreaKHis')
    ap.add_argument('--output', type=Path, default=Path('resultados_breakhis'))
    ap.add_argument('--reference', type=Path, help='Imagen de referencia; elegir de entrenamiento')
    ap.add_argument('--method', choices=['none', 'reinhard', 'macenko', 'vahadane', 'all'], default='all')
    ap.add_argument('--limit', type=int, default=8, help='Cantidad máxima de imágenes procesadas; 0=todas')
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--min-area', type=int, default=35)
    args = ap.parse_args()
    if not args.dataset.is_dir():
        ap.error('La carpeta --dataset no existe')
    files = sorted(p for p in args.dataset.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    rows = [m for p in files if (m := metadata(p)) is not None]
    if not rows:
        ap.error('No se identificaron imágenes 40X/100X. Revisa nombres y carpetas.')
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for row in rows:
        try:
            img = read_rgb(Path(row['ruta']))
            h, w = img.shape[:2]
            inventory.append({**row, 'ancho': w, 'alto': h, 'estado': 'ok'})
        except Exception as e:
            inventory.append({**row, 'ancho': None, 'alto': None, 'estado': f'error: {e}'})
    df = pd.DataFrame(inventory)
    df.to_csv(args.output / 'inventario_40x_100x.csv', index=False, encoding='utf-8-sig')
    counts = pd.crosstab(df['magnificacion'], df['clase'])
    counts.to_csv(args.output / 'conteos.csv', encoding='utf-8-sig')
    print('\nINVENTARIO BreaKHis 40X y 100X:\n', counts.to_string())
    valid = df[df['estado'].eq('ok')]
    if valid.empty:
        raise RuntimeError('No hay imágenes legibles')
    if args.reference:
        refpath = args.reference
    else:
        refpath = Path(valid.iloc[0]['ruta'])
        warnings.warn('No se especificó --reference; se usa la primera imagen legible SOLO '
                      'para una demostración. Para experimentos, seleccionar referencia en train.')
    reference = read_rgb(refpath)
    methods = ['none', 'reinhard', 'macenko', 'vahadane'] if args.method == 'all' else [args.method]
    target_mean, target_std = reinhard_stats(reference)
    mac_target = None
    vah_target = None
    if 'macenko' in methods:
        try:
            mac_target = macenko_fit(reference)
        except Exception as e:
            print('ADVERTENCIA Macenko no disponible:', e)
    if 'vahadane' in methods:
        try:
            vah_target = vahadane_normalizer(reference)
        except Exception as e:
            print('ADVERTENCIA Vahadane omitido:', e)
    sample = valid if args.limit == 0 else valid.head(max(0, args.limit))
    out_metrics = []
    for _, row in sample.iterrows():
        original = read_rgb(Path(row['ruta']))
        for method in methods:
            try:
                if method == 'none':
                    transformed = original
                elif method == 'reinhard':
                    transformed = reinhard_transform(original, target_mean, target_std)
                elif method == 'macenko':
                    if mac_target is None:
                        continue
                    transformed = macenko_transform(original, *mac_target)
                else:
                    if vah_target is None:
                        continue
                    transformed = vah_target.transform(original)
                maps = analyze(transformed, args.kernel, args.min_area)
                stem = Path(row['archivo']).stem
                dest = args.output / method / row['magnificacion'] / stem
                write_rgb(dest / 'normalizada.png', transformed)
                for key, val in maps.items():
                    save_gray(dest / f'{key}.png', val)
                save_montage(dest / 'comparacion.png', original, transformed, maps)
                out_metrics.append({'archivo': row['archivo'], 'clase': row['clase'],
                                    'magnificacion': row['magnificacion'], 'metodo': method,
                                    'porcentaje_mascara_hsv': float(np.mean(maps['limpia_hsv'] > 0) * 100),
                                    'porcentaje_mascara_lab': float(np.mean(maps['limpia_lab'] > 0) * 100)})
                print(f"OK: {row['archivo']} / {method}")
            except Exception as e:
                print(f"ERROR: {row['archivo']} / {method}: {e}")
    pd.DataFrame(out_metrics, columns=['archivo', 'clase', 'magnificacion', 'metodo',
          'porcentaje_mascara_hsv', 'porcentaje_mascara_lab']).to_csv(
              args.output / 'resumen_mascaras.csv', index=False, encoding='utf-8-sig')
    print(f'\nResultados guardados en: {args.output.resolve()}')

if __name__ == '__main__':
    main()
