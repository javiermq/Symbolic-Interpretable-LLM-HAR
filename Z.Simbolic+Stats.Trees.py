STATS = True   # ← activa para usar estadísticas
LOGIC = True   # ← activa para usar lógica difusa

labels_L5 = ["is Very low", "is Low", "is Medium", "is High", "is Very high"]
labels_L2 = ["is Enable", "is Disabled"]

labels_Q5 = ["None", "Few", "Some", "Many", "All"]

BINARY_TYPES = {"binary"}

points = [0.0, 0.25, 0.5, 0.75, 1.0]

WINDOW_SIZE_ROWS = 128

THRESH_FEATURE = 0.5


USE_QUANTIZE = True     # ⬅️ pon False si no quieres cuantizar
Q_STEP = 0.05          # granularidad (0.25 → {0, 0.25, 0.5, 0.75, 1.0})

USE_QUANTIZE2=True
Q_STEP2 = 0.01

T5=5
labels_T5 = [
    "at the beginning",      
    "early in the segment",  
    "in the middle",         
    "toward the end",       
    "at the end"           
]

SW=5
NTrees=25

LABEL_KEYS = ["activity", "label", "activity_name"]
BAD_LABELS = {"none", "unknown", "transition", "TRANSITION"}



import numpy as np
import pandas as pd
import re
from copy import deepcopy
import yaml
import argparse, os
ap = argparse.ArgumentParser(description="Run fuzzy_terms pipeline (converted from notebook)")
ap.add_argument("--out", dest="out_dir", default="output",
                help="Main output folder for figures, logs, etc.")
ap.add_argument("--FUZZY_DATA_OUT", dest="data_out_dir", default="data/mhealth",
                help="Folder for final dataset files (train2.jsonl, val2.jsonl)")
# 👇 nuevo
ap.add_argument("--train_jsonl", dest="train_jsonl", required=True,
                help="Ruta al jsonl de ventanas de train")
ap.add_argument("--test_jsonl", dest="test_jsonl", required=True,
                help="Ruta al jsonl de ventanas de test")
ap.add_argument(
    "--sensor_cfg",
    dest="sensor_cfg",
    default=None,
    help="Ruta a YAML con definición de sensores (sensors: {subtipo: [canales]})"
)

args = ap.parse_args()

globals()["FUZZY_OUT"] = args.out_dir
globals()["FUZZY_DATA_OUT"] = args.data_out_dir
globals()["FUZZY_TRAIN_JSONL"] = args.train_jsonl
globals()["FUZZY_TEST_JSONL"] = args.test_jsonl

os.environ["FUZZY_OUT"] = args.out_dir
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.data_out_dir, exist_ok=True)

print(f"[fuzzy_terms] Output dir:     {args.out_dir}")
print(f"[fuzzy_terms] Data out dir:   {args.data_out_dir}")
print(f"[fuzzy_terms] Train jsonl:    {args.train_jsonl}")
print(f"[fuzzy_terms] Test jsonl:     {args.test_jsonl}")
print(f"[fuzzy_terms] Sensor cfg:     {args.sensor_cfg}")
                
                


globals()["FUZZY_OUT"] = args.out_dir
globals()["FUZZY_DATA_OUT"] = args.data_out_dir
# 👇 hacer visibles las rutas para la sección READ INPUT DATA
globals()["FUZZY_TRAIN_JSONL"] = args.train_jsonl
globals()["FUZZY_TEST_JSONL"] = args.test_jsonl

os.environ["FUZZY_OUT"] = args.out_dir


os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.data_out_dir, exist_ok=True)


print(f"[fuzzy_terms] Output dir:     {args.out_dir}")
print(f"[fuzzy_terms] Data out dir:   {args.data_out_dir}")
print(f"[fuzzy_terms] Train jsonl:    {args.train_jsonl}")
print(f"[fuzzy_terms] Test jsonl:     {args.test_jsonl}")

# =========================
# Cargar definición de sensores desde YAML (opcional)
# =========================
CHANNEL_TYPE_MAP = {}

if args.sensor_cfg is not None:
    with open(args.sensor_cfg, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    sensors_cfg = cfg_yaml.get("sensors", {})

    if not isinstance(sensors_cfg, dict):
        raise ValueError("[sensor_cfg] La clave 'sensors' debe ser un dict subtipo -> [canales]")

    for subtype, chans in sensors_cfg.items():
        if chans is None:
            continue
        if not isinstance(chans, (list, tuple)):
            raise ValueError(f"[sensor_cfg] El grupo '{subtype}' debe ser una lista de nombres de canal")
        for ch in chans:
            if ch in CHANNEL_TYPE_MAP and CHANNEL_TYPE_MAP[ch] != subtype:
                print(f"[sensor_cfg][WARN] Canal '{ch}' aparece en dos subtipos: "
                      f"{CHANNEL_TYPE_MAP[ch]} y {subtype}")
            CHANNEL_TYPE_MAP[ch] = subtype

    print("[sensor_cfg] Mapa canal→subtipo cargado:")
    for ch, tp in sorted(CHANNEL_TYPE_MAP.items()):
        print(f"  {ch:35s} → {tp}")
else:
    print("[sensor_cfg] No se ha pasado --sensor_cfg, se usará inferencia heurística")

# ================================================
# Construcción dinámica de WEAR_SENSORS y BINARY_SENSORS
# ================================================

WEAR_SENSORS = []
BINARY_SENSORS = []

if CHANNEL_TYPE_MAP:
    for ch, tp in CHANNEL_TYPE_MAP.items():
        tp_upper = tp.upper()

        if tp_upper == "WEAR":
            WEAR_SENSORS.append(ch)

        elif tp_upper == "BINARY":
            BINARY_SENSORS.append(ch)

        else:
            raise ValueError(
                f"[sensor_cfg] Tipo '{tp}' no permitido. "
                f"Solo se aceptan WEAR y BINARY. Canal: {ch}"
            )
else:
    raise ValueError(
        "[sensor_cfg] No hay CHANNEL_TYPE_MAP cargado. "
        "Debes pasar --sensor_cfg con un YAML válido."
    )

print("\n[SENSORS] WEAR_SENSORS dinámicos:")
for s in WEAR_SENSORS:
    print("   ", s)

print("\n[SENSORS] BINARY_SENSORS dinámicos:")
for s in BINARY_SENSORS:
    print("   ", s)

   

import json
import numpy as np
import os

TRAIN_JSONL_PATH = globals().get("FUZZY_TRAIN_JSONL", None)
TEST_JSONL_PATH  = globals().get("FUZZY_TEST_JSONL", None)

def _read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

if TRAIN_JSONL_PATH is None or TEST_JSONL_PATH is None:
    raise RuntimeError("Debes pasar --train_jsonl y --test_jsonl al script para leer las ventanas.")

train_records = _read_jsonl(TRAIN_JSONL_PATH)
test_records  = _read_jsonl(TEST_JSONL_PATH)

# =========================
# Convertimos a segmentos numéricos
# =========================
# asumimos que todos los 'window' tienen exactamente las mismas keys y mismo largo
example_rec = train_records[0]
example_win = example_rec["window"]

def window_to_array_generic(rec):
    """
    Soporta:
      1) Formato antiguo: rec["window"] = {canal: [T]}
      2) Formato nuevo:   rec["channels"] = [canales], rec["window"] = matriz
    Devuelve:
        arr: (T, F)
        sensor_cols: lista de nombres de canales
    """
    win = rec["window"]

    # --- Formato nuevo SPHERE/MARBLE ---
    if "channels" in rec and isinstance(win, list):
        sensor_cols = list(rec["channels"])
        arr = np.asarray(win, dtype=float)

        if arr.ndim != 2:
            raise ValueError(f"window ndim={arr.ndim}, esperaba 2")

        C = len(sensor_cols)
        if arr.shape[0] == C and arr.shape[1] != C:
            arr = arr.T             # (F,T) -> (T,F)
        elif arr.shape[1] == C:
            pass                    # ya (T,F)
        else:
            raise ValueError(f"Inconsistencia channels={C}, window.shape={arr.shape}")
        arr = np.round(arr, decimals=2)
        return arr, sensor_cols

    # --- Formato antiguo: dict {canal: [T]} ---
    if isinstance(win, dict):
        sensor_cols = list(win.keys())
        arr_cols = [np.asarray(win[c], dtype=float) for c in sensor_cols]
        arr = np.stack(arr_cols, axis=1)
        arr = np.round(arr, decimals=2)
        return arr, sensor_cols

    raise ValueError("Formato de 'window' no reconocido.")


# Detecta columnas y tamaño temporal
example_arr, sensor_cols = window_to_array_generic(example_rec)
T = example_arr.shape[0]
WINDOW_SIZE_ROWS = T
print("[fuzzy] WINDOW_SIZE_ROWS:", WINDOW_SIZE_ROWS)
print("[fuzzy] sensor_cols:", sensor_cols)



def infer_channel_type(ch_name: str) -> str:
    """
    Devuelve el subtipo de canal.

    - Si hay YAML (--sensor_cfg), se usa exclusivamente CHANNEL_TYPE_MAP
      y se lanza error si el canal no está definido.
    - Si NO hay YAML, se cae al comportamiento heurístico anterior.
    """
    # 1) Si tenemos YAML, usarlo SÍ o SÍ
    global CHANNEL_TYPE_MAP
    if CHANNEL_TYPE_MAP:
        if ch_name not in CHANNEL_TYPE_MAP:
            raise ValueError(
                f"[sensor_cfg] Canal '{ch_name}' no definido en la sección 'sensors' del YAML. "
                f"Añádelo a algún subtipo (p.ej. location/binary/imu/...)."
            )
        return CHANNEL_TYPE_MAP[ch_name]

    # 2) Sin YAML: heurística antigua
    # NUEVO DATASET: prefijos loc_ y obj_
    if ch_name.startswith("loc_"):
        return "location"
    if ch_name.startswith("obj_"):
        return "binary"

    # Compatibilidad MARBLE/MHEALTH etc.
    if ch_name.startswith("env_") or ch_name.startswith("smartphone_"):
        return "binary"
    if ch_name.isupper() and "_" in ch_name:
        return "location"
    if ch_name.startswith("accelerometer_") or ch_name.startswith("gyroscope_") \
       or ch_name.startswith("magnetometer_"):
        return "imu"

    return "other"


def quantize(X, step=0.1):
    """Quantiza X al paso dado y garantiza rango [0, 1]."""
    Xq = np.round(X / step) * step
    return np.clip(Xq, 0.0, 1.0).astype(np.float32)

# =========================
# Convertimos TODOS los registros a arrays (T, F)
# =========================
def _normalize_label(x):
    # Normaliza etiquetas a string "unknown" si vienen como None o vacías
    if not isinstance(x, str) or not x.strip():
        return "unknown"
    return x.strip()

all_train_segments = []
all_train_labels   = []



def get_label_from_record(rec):
    """
    Busca la etiqueta en varios campos típicos para que funcione
    tanto con UWB (activity) como con MARBLE (label / activity_name).
    """
    for k in LABEL_KEYS:
        if k in rec and isinstance(rec[k], str) and rec[k].strip():
            return rec[k]
    return None

train_records_kept = []
for i, rec in enumerate(train_records):
    seg, _ = window_to_array_generic(rec)

    raw_label = get_label_from_record(rec)
    lbl = _normalize_label(raw_label)

    if lbl in BAD_LABELS:
        print("unkonw", {
            "label": lbl,
            "timestamp": rec.get("timestamp", None),
            "user_label": rec.get("user_label", rec.get("description", None)),
            "user_id": rec.get("user_id", rec.get("subject", None)),
            "window_shape": np.asarray(rec["window"]).shape if "window" in rec else None,
        })
        continue
        

    all_train_segments.append(seg)
    all_train_labels.append({
        "label": lbl,
        "timestamp": rec.get("timestamp", None),
        "user_label": rec.get("user_label", None),
        "user_id": rec.get("user_id", None),
    })
    train_records_kept.append(rec)

all_test_segments = []
all_test_labels   = []

test_records_kept = []
for rec in test_records:
    seg, _ = window_to_array_generic(rec)

    raw_label = get_label_from_record(rec)
    lbl = _normalize_label(raw_label)

    if lbl in BAD_LABELS:
        print("unkonw", {
            "label": lbl,
            "timestamp": rec.get("timestamp", None),
            "user_label": rec.get("user_label", rec.get("description", None)),
            "user_id": rec.get("user_id", rec.get("subject", None)),
            "window_shape": np.asarray(rec["window"]).shape if "window" in rec else None,
        })
        continue

    all_test_segments.append(seg)
    all_test_labels.append({
        "label": lbl,
        "timestamp": rec.get("timestamp", None),
        "user_label": rec.get("user_label", None),
        "user_id": rec.get("user_id", None),
    })
    test_records_kept.append(rec)



print(f"[fuzzy] train windows leídos: {len(all_train_segments)} | test windows: {len(all_test_segments)}")
print(f"[fuzzy] columnas de sensores: {sensor_cols}")
print(f"[fuzzy] largo temporal detectado: {WINDOW_SIZE_ROWS}")

                        

def trapezoidal_membership_array(x, a, b, c, d):
    """
    μ(x) trapezoidal vectorizada.
    Casos especiales: triángulos (a=b o c=d) y punto (a=b=c=d).
    """
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x, dtype=float)

    # Punto singular
    if a == b == c == d:
        mu = (x == a).astype(float)
        return mu

    # a < x < b (pendiente izq.)
    mask = (x > a) & (x < b)
    mu[mask] = (x[mask] - a) / (b - a) if b != a else 1.0

    # b <= x <= c (meseta)
    mask = (x >= b) & (x <= c)
    mu[mask] = 1.0

    # c < x < d (pendiente der.)
    mask = (x > c) & (x < d)
    mu[mask] = (d - x[mask]) / (d - c) if d != c else 1.0

    # fuera: 0
    return mu

def build_trapezoidal_partition(vmin, vmax, L=5, labels=None,
                                      plateau_ratio=0.6, overlap_ratio=0.3):
    """
    Partición trapezoidal cruzada en [vmin, vmax].

    - centros equiespaciados
    - soportes solapados controlados por overlap_ratio
    - mesetas (b..c) grandes (plateau_ratio del soporte)
    - siempre hay al menos una función con μ=1 en todo el rango
    """
    if labels is None:
        labels = [f"L{i+1}" for i in range(L)]

    if vmax <= vmin or L < 2:
        mid = float(vmin)
        return [{"label": labels[i], "a": mid, "b": mid, "c": mid, "d": mid} for i in range(L)]

    centers = np.linspace(vmin, vmax, L)
    step = centers[1] - centers[0]

    parts = []
    for i in range(L):
        if i == 0:
            # Triangular izquierdo con meseta que arranca en vmin
            midR = (centers[i] + centers[i+1]) / 2.0
            a = vmin
            d = midR + overlap_ratio * step
            support = d - a
            plateau = plateau_ratio * support
            b = a                  # meseta desde el borde izquierdo
            c = min(d, a + plateau)
        elif i == L - 1:
            # Triangular derecho con meseta que acaba en vmax
            midL = (centers[i-1] + centers[i]) / 2.0
            a = midL - overlap_ratio * step
            d = vmax
            support = d - a
            plateau = plateau_ratio * support
            c = d                  # meseta hasta el borde derecho
            b = max(a, d - plateau)
        else:
            # Trapezoide intermedio con soporte ampliado para cruzarse
            midL = (centers[i-1] + centers[i]) / 2.0
            midR = (centers[i]   + centers[i+1]) / 2.0
            a = midL - overlap_ratio * step
            d = midR + overlap_ratio * step
            a = max(a, vmin)
            d = min(d, vmax)
            support = d - a
            plateau = plateau_ratio * support
            b = centers[i] - plateau/2
            c = centers[i] + plateau/2
            b = max(a, b)
            c = min(d, c)

        parts.append({
            "label": labels[i],
            "a": float(a), "b": float(b), "c": float(c), "d": float(d)
        })
    return parts






# ==== Cell 14 ====

# ==== Cell 15 ====
def apply_fuzzy_segments(segments, sensor_cols, linguistic_scales):
    """
    segments: lista de arrays (T, F) con F = len(sensor_cols)
    linguistic_scales: dict {col -> lista de dicts {a,b,c,d,label}}
                       OJO: cada col puede tener un nº distinto de labels (L2 ó L5).
    Devuelve:
      - lista de arrays fuzzificados (T, sum_L)
      - lista de nombres de columnas en el orden de concatenación
    """
    fuzzy_segments = []
    out_cols = None  # la rellenamos usando el primer segmento

    for seg in segments:
        T, F = seg.shape
        assert F == len(sensor_cols), f"Segment features={F} no coincide con sensor_cols={len(sensor_cols)}"

        per_sensor_blocks = []
        local_out_cols = []

        for j, col in enumerate(sensor_cols):
            x = seg[:, j]  # (T,)
            params_list = linguistic_scales[col]  # lista de dicts {a,b,c,d,label}
            L_col = len(params_list)

            memb = np.zeros((T, L_col), dtype=float)
            for k, p in enumerate(params_list):
                a, b, c, d = p["a"], p["b"], p["c"], p["d"]
                memb[:, k] = trapezoidal_membership_array(x, a, b, c, d)
                # nombres de columnas para este canal/label
                local_out_cols.append(f"{col} [{p['label']}]")

            per_sensor_blocks.append(memb)

        seg_fuzzy = np.hstack(per_sensor_blocks)  # (T, sum_L)
        fuzzy_segments.append(seg_fuzzy)

        if out_cols is None:
            out_cols = local_out_cols

    return fuzzy_segments, out_cols

import pandas as pd
import numpy as np

# all_train_segments: lista de (T, F)
all_train_data = np.stack(all_train_segments, axis=0)  # (N, T, F)
N, T, F = all_train_data.shape

# aplanamos N y T
all_train_flat = all_train_data.reshape(N * T, F)      # (N*T, F)

# ⚠️ usar nanmin / nanmax porque hay NaN en los datos
mins = np.nanmin(all_train_flat, axis=0)
maxs = np.nanmax(all_train_flat, axis=0)

# por si alguna columna es todo NaN, la rellenamos con 0..1
for i, (mn, mx) in enumerate(zip(mins, maxs)):
    if np.isnan(mn) and np.isnan(mx):
        mins[i] = 0.0
        maxs[i] = 1.0
    elif np.isnan(mn):
        mins[i] = maxs[i] - 1.0
    elif np.isnan(mx):
        maxs[i] = mins[i] + 1.0

col_min_train = pd.Series(mins, index=sensor_cols)
col_max_train = pd.Series(maxs, index=sensor_cols)

print("col_min_train", col_min_train)
print("col_max_train", col_max_train)

linguistic_scales = {}

# channel_types = lista paralela a sensor_cols
channel_types = [infer_channel_type(ch) for ch in sensor_cols]

print("[fuzzy] canal → tipo (dinámico):")
for ch, tp in zip(sensor_cols, channel_types):
    print(f"  {ch:35s} → {tp}")

# mapa usable por el resto del pipeline
ch2type = dict(zip(sensor_cols, channel_types))



for col in sensor_cols:
    ctype = ch2type.get(col, "other")

    if ctype in BINARY_TYPES:
        # Binarios (rooms + env + smartphone) → L2 en [0,1]
        print(ctype, "IS ", BINARY_TYPES)
        vmin, vmax = 0.0, 1.0
        linguistic_scales[col] = build_trapezoidal_partition(
            vmin, vmax,
            L=2,
            labels=labels_L2,
        )
    else:
        # Continuos (imu, etc.) → L5 en [vmin, vmax] del train
        vmin = float(col_min_train[col])
        vmax = float(col_max_train[col])
        if vmax <= vmin:
            vmax = vmin + 1e-6

        linguistic_scales[col] = build_trapezoidal_partition(
            vmin, vmax,
            L=5,
            labels=labels_L5,
             plateau_ratio=0.6,   # más grande ⇒ meseta más ancha
            overlap_ratio=0.3    # más grande ⇒ más cruce entre funciones            
        )

    print(col, ctype, "→", linguistic_scales[col])



# verificación rápida
#print("Ventanas train:", len(all_train_segments),
#      "| shape fuzzy[0]:", train_segments_fuzzy[0].shape if len(train_segments_fuzzy) else None)
#print("Ventanas test:", len(all_test_segments),
#      "| shape fuzzy[0]:", test_segments_fuzzy[0].shape if len(test_segments_fuzzy) else None)


# ==== Cell 16 ====
def build_temporal_partition_trapezoidal_cross(
    vmin, vmax, L=5, labels=None,
    plateau_ratio=0.6,
    overlap_ratio=0.3,
):
    """
    Partición temporal cruzada en [vmin, vmax], reutilizando la misma
    lógica que las escalas lingüísticas (build_trapezoidal_partition).

    vmin, vmax: índice temporal (ej. 0, T-1)
    L: número de escalas temporales (ej. T5=5)
    labels: labels_T5, p.ej. ["at the beginning", ..., "at the end"]
    """
    return build_trapezoidal_partition(
        vmin=vmin,
        vmax=vmax,
        L=L,
        labels=labels,
        plateau_ratio=plateau_ratio,
        overlap_ratio=overlap_ratio,
    )




t_idx = np.arange(WINDOW_SIZE_ROWS, dtype=float)

vmin, vmax = float(t_idx.min()), float(t_idx.max())
temporal_scales = build_temporal_partition_trapezoidal_cross(vmin, vmax, L=T5, labels=labels_T5)

# ---- Iterar sobre t y calcular μ para cada escala ----
rows = []
for t in t_idx.astype(int):
    row = {"t": t}
    for p in temporal_scales:
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
        lbl = p["label"]
        # calculamos μ(t) (array de 1 elemento → tomamos [0])
        row[lbl] = trapezoidal_membership_array(np.array([float(t)]), a, b, c, d)[0]
    rows.append(row)

temporal_fuzzy_df = pd.DataFrame(rows).set_index("t")

# Ejemplos de inspección
print("Índice temporal (primeros 10):", temporal_fuzzy_df.index[:10].to_numpy())
print("Parámetros de las escalas:")
print(pd.DataFrame(temporal_scales))

# ==== Cell 17 ====
def temporal_weighted_aggregate(
    segments_fuzzy,
    temporal_fuzzy_df,
    temporal_labels,
    fuzzy_feature_names,
    keep_3d=False
):
    """
    Agregación ponderada temporal de segmentos fuzzificados.

    Args:
        segments_fuzzy: lista de arrays (T, D)
        temporal_fuzzy_df: DataFrame (T, L_temporal) con μ temporales
        temporal_labels: nombres de escalas temporales (labels_T5)
        fuzzy_feature_names: nombres base de D características difusas (sensor + label lingüístico)
        keep_3d: si True => (N, L_temporal, D), si False => (N, L_temporal*D)

    Returns:
        agg_array: np.ndarray
        out_feature_names: lista de nombres combinados (fuzzy + temporal)
    """
    weights = [temporal_fuzzy_df[lbl].values.astype(float) for lbl in temporal_labels]
    T_ref = len(temporal_fuzzy_df)
    agg_list = []

    for seg in segments_fuzzy:
        T, D = seg.shape
        assert T == T_ref, f"Longitud temporal ({T}) != {T_ref}"
        per_tau = []
        for w in weights:
            s = w.sum()
            if s == 0:
                w_norm = np.ones_like(w) / len(w)
                agg_tau = (w_norm[:, None] * seg).sum(axis=0)
            else:
                agg_tau = (w[:, None] * seg).sum(axis=0) / s
            per_tau.append(agg_tau[None, :])  # (1, D)
        per_tau = np.vstack(per_tau)  # (L_temporal, D)
        agg_list.append(per_tau)

    agg_array = np.stack(agg_list, axis=0)  # (N, L_temporal, D)

    if keep_3d:
        return agg_array, None

    # Aplanar
    N, Ltemp, D = agg_array.shape
    agg_flat = agg_array.reshape(N, Ltemp * D)

    # 🔹 Unificar nombres: fuzzy_feature_name + " " + temporal_label
    out_feature_names = []
    for tau in temporal_labels:
        for name in fuzzy_feature_names:
            out_feature_names.append(f"{name} [{tau}]")  # espacio entre ambos labels

    return agg_flat, out_feature_names






# ==== Cell 19 ====
# --- (1) Construcción de cuantificadores y membresía (tu versión) ---
def build_fuzzy_quantifiers_from_points(
    points,
    labels=None,
    plateau_ratio=0.6,
    overlap_ratio=0.3,
):
    """
    Construye cuantificadores difusos (None, Few, Some, Many, All)
    sobre el dominio [0,1] usando la misma partición cruzada que L5.

    'points' se usa solo para saber cuántos cuantificadores (L).
    Se asume que cubren [0,1] razonablemente; con [0, .25, .5, .75, 1]
    equivale a una partición uniforme.
    """
    if labels is None:
        labels = ["None", "Few", "Some", "Many", "All"]

    L = len(points)
    assert L == len(labels), "Debe haber el mismo número de puntos y etiquetas"

    # Reutilizamos la misma lógica cruzada: dominio fijo [0,1]
    # (forma invariante, sólo importa L y labels).
    return build_trapezoidal_partition(
        vmin=0.0,
        vmax=1.0,
        L=L,
        labels=labels,
        plateau_ratio=plateau_ratio,
        overlap_ratio=overlap_ratio,
    )


def Q_trapezoidal_membership(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    # subida
    left  = np.where((x > a) & (x < b), (x - a) / (b - a + 1e-12), 0.0)
    # meseta
    mid   = np.where((x >= b) & (x <= c), 1.0, 0.0)
    # bajada
    right = np.where((x > c) & (x < d), (d - x) / (d - c + 1e-12), 0.0)
    mu = left + mid + right
    mu[(x == b) | (x == c)] = 1.0
    return np.clip(mu, 0.0, 1.0)

# --- (2) Aplicación a TODAS las features: expande D -> D*5 ---
def apply_quantifiers_to_features(X, feature_names, quantifiers):
    """
    X: (N, D) en [0,1]
    feature_names: lista de D nombres
    quantifiers: lista de dicts {label,a,b,c,d}
    Return:
      Xq: (N, D*Q)
      names_q: lista de D*Q nombres "<feature> | <quantifier>"
    """
    N, D = X.shape
    Q = len(quantifiers)

    # por seguridad, recorta a [0,1]
    X = np.clip(X, 0.0, 1.0)

    blocks = []
    names_q = []
    for q in quantifiers:
        a, b, c, d = q["a"], q["b"], q["c"], q["d"]
        mu = Q_trapezoidal_membership(X, a, b, c, d)  # vectorizada sobre toda la matriz
        blocks.append(mu)  # (N, D)
        names_q.extend([f"[{q['label']}] {fname} " for fname in feature_names])

    Xq = np.hstack(blocks)  # (N, D*Q)
    return Xq, names_q


if LOGIC:
    # ============================
    #  Pipeline LÓGICO / FUZZY
    # ============================
    train_segments_fuzzy, fuzzy_feature_names = apply_fuzzy_segments(
        all_train_segments, sensor_cols, linguistic_scales
    )
    test_segments_fuzzy, _ = apply_fuzzy_segments(
        all_test_segments, sensor_cols, linguistic_scales
    )

    train_windows_features, train_feat_names = temporal_weighted_aggregate(
        train_segments_fuzzy,
        temporal_fuzzy_df,
        labels_T5,
        fuzzy_feature_names=fuzzy_feature_names,
        keep_3d=False
    )
    test_windows_features, test_feat_names = temporal_weighted_aggregate(
        test_segments_fuzzy,
        temporal_fuzzy_df,
        labels_T5,
        fuzzy_feature_names=fuzzy_feature_names,
        keep_3d=False
    )

    # Quantifiers
    quantifiers = build_fuzzy_quantifiers_from_points(points, labels_Q5)

    train_windows_features_q, train_feat_names_q = apply_quantifiers_to_features(
        train_windows_features, train_feat_names, quantifiers
    )
    test_windows_features_q,  test_feat_names_q  = apply_quantifiers_to_features(
        test_windows_features,  test_feat_names,  quantifiers
    )

    # 👇 CUANTIZAR SOLO LAS FEATURES LÓGICAS (en [0,1])
    if USE_QUANTIZE:
        X_logic_train = quantize(train_windows_features_q, Q_STEP)
        X_logic_test  = quantize(test_windows_features_q,  Q_STEP)
    else:
        X_logic_train = train_windows_features_q
        X_logic_test  = test_windows_features_q

    print("Original train shape:", train_windows_features.shape)
    print("Expanded  train shape:", train_windows_features_q.shape)
    print("Original test  shape:", test_windows_features.shape)
    print("Expanded  test  shape:", test_windows_features_q.shape)
    print("\nEjemplo de nombres de columnas expandidas:")
    print(train_feat_names_q[:5])

else:
    train_windows_features_q = None
    test_windows_features_q  = None
    train_feat_names_q = None
    X_logic_train = None
    X_logic_test  = None


# --------------------------
# (5) Comprobaciones
# --------------------------



# ==== Cell 21 ====
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC





# ============================================================
# Construir índices automáticos de etiquetas (label → int)
# ============================================================
# recolectamos todas las etiquetas únicas del train y test
all_labels_text = sorted(set([lab["label"] for lab in all_train_labels] +
                             [lab["label"] for lab in all_test_labels]))

# asignamos un ID incremental a cada etiqueta
label2id = {lbl: idx for idx, lbl in enumerate(all_labels_text)}
id2label = {idx: lbl for lbl, idx in label2id.items()}

activity_map = {i + 1: lbl for i, lbl in enumerate(all_labels_text)}

print("[fuzzy] activity_map generado dinámicamente:")
for k, v in activity_map.items():
    print(f"  {k}: {v}")
    
print("[fuzzy] Etiquetas detectadas:", label2id)

# creamos los arrays numéricos usando los índices auto-generados
y_train = np.array([label2id[lab["label"]] for lab in all_train_labels], dtype=int)
y_test  = np.array([label2id[lab["label"]] for lab in all_test_labels], dtype=int)

num_labels = len(label2id)
print(f"[fuzzy] num_labels={num_labels}")



    
# =========================
# Features statis
# =========================    

def quantize2(X, step=0.1):
    """
    Quantiza X al paso dado, SIN reescalar y SIN recortar a [0,1].
    -9.154 -> -9.2 (si step=0.1)
    0.037  -> 0.0
    """
    Xq = np.round(X / step) * step
    return Xq.astype(np.float32)
    
    
import math

import numpy as np

def stats_from_segment_nanaware(seg_np: np.ndarray):
    """
    Calcula mean, std, min, max y porcentaje de ceros por canal de forma NaN-aware.

    Devuelve:
        stats: array (5, F) con filas:
               [0] -> mean   (sobre valores no-NaN)
               [1] -> std    (sobre valores no-NaN)
               [2] -> min    (sobre valores no-NaN)
               [3] -> max    (sobre valores no-NaN)
               [4] -> void   (porcentaje de ceros en el segmento, ignorando NaN)
    """
    T, F = seg_np.shape

    means = np.zeros(F, dtype=float)
    stds  = np.zeros(F, dtype=float)
    mins  = np.zeros(F, dtype=float)
    maxs  = np.zeros(F, dtype=float)
    voids = np.zeros(F, dtype=float)

    for j in range(F):
        col = seg_np[:, j]
        valid = ~np.isnan(col)
        v = col[valid]

        if v.size == 0:
            # si no hay valores válidos: dejamos todo a 0.0 y void=1.0
            means[j] = 0.0
            stds[j]  = 0.0
            mins[j]  = 0.0
            maxs[j]  = 0.0
            voids[j] = 1.0
            continue

        means[j] = float(np.median(v))
        stds[j]  = float(np.std(v, ddof=0))
        mins[j]  = float(np.min(v))
        maxs[j]  = float(np.max(v))

        # porcentaje de minimos entre los valores válidos
        zero_count = np.sum(v == 0.0)
        voids[j]   = float(zero_count) / float(v.size)

    stats = np.stack([means, stds, mins, maxs, voids], axis=0)  # (5, F)
    return stats


def _is_binary_type(tp: str) -> bool:
    """
    Determina si un tipo de canal debe tratarse como binario.
    Soporta tipos del YAML (BINARY) y heurísticos (binary, location, env, smartphone).
    Lanza una excepción si el tipo no es 'BINARY' ni uno de los tipos heurísticos.
    """
    if tp is None:
        return False
    
    tp_str = str(tp)
    tp_low = tp_str.lower()
    tp_up  = tp_str.upper()

    # 1. Tipo YAML (BINARY)
    if tp_up == "BINARY":
        return True

    
    if tp_low in {"binary"}:
        return True
    return False


def _is_wear_type(tp: str) -> bool:
    """
    Determina si un tipo de canal es wearable / continuo (IMU).
    Soporta tipos del YAML (WEAR) y heurísticos (imu).
    """
    if tp is None:
        return False
    tp_low = str(tp).lower()
    tp_up  = str(tp).upper()

    if tp_up == "WEAR":
        return True

    # Heurística cuando no hay YAML
    if tp_low in {"imu", "wear"}:
        return True
    return False

def extract_subwindow_features(
    win_arr,
    sensor_cols,
    channel_types=None,
    n_parts=SW,
    extra_group_sizes=(2, 3),
    void_descriptions_out=None,  # ya no se usa, pero lo dejamos en la firma por compatibilidad
):
    """
    win_arr: np.array (T, F)
    sensor_cols: lista de nombres de canales, len=F
    channel_types: lista paralela a sensor_cols con el tipo de canal
                   (p.ej. ['WEAR','WEAR','BINARY',...]).
                   Si es None, se infiere con infer_channel_type().
    n_parts: dividir el eje temporal en este nº de partes
    extra_group_sizes: tamaños adicionales de grupos de partes

    Devuelve:
        feats_flat: vector 1D con todas las features
        feat_names: lista de nombres en mismo orden

    Estadísticas generadas por canal y por subventana/grupo:
      - mean, std, min, max, void   (todas las columnas)
      - last_value_bin, n_changes_bin   (solo para canales binarios; 0.0 resto)
      - skewness_wear, kurtosis_wear    (solo para wearables; 0.0 resto)
    """
    win_arr = np.asarray(win_arr, dtype=float)
    T, F = win_arr.shape

    # Si no nos pasan channel_types, los inferimos
    if channel_types is None:
        channel_types = [infer_channel_type(ch) for ch in sensor_cols]
    else:
        # por seguridad, aseguramos longitud
        assert len(channel_types) == len(sensor_cols), \
            f"channel_types len={len(channel_types)} != len(sensor_cols)={len(sensor_cols)}"

    part_len = T // n_parts

    # 1) segmentar ventana en partes básicas
    parts = []
    for p in range(n_parts):
        start = p * part_len
        end = (p + 1) * part_len if p < n_parts - 1 else T
        seg = win_arr[start:end, :]      # (L_p, F)
        parts.append(seg)

    all_feats = []
    all_names = []

    # stats base que ya tenías
    stats_names = ["mean", "desviation", "min", "max", "void"]

    # -------------------------------------------------
    # 2) features de cada parte por separado: p0, p1, ...
    # -------------------------------------------------
    for p, seg in enumerate(parts):
        # (5, F): median (mean), std, min, max, void
        stats_this_part = stats_from_segment_nanaware(seg)

        # --- 2.1) estadísticas base (todas las columnas) ---
        for stat_name, stat_values in zip(stats_names, stats_this_part):
            all_feats.append(stat_values)  # (F,)
            for ch in sensor_cols:
                all_names.append(f"#p{p}@{stat_name}@{ch}")

        # --- 2.2) estadísticas específicas por tipo ---
                # --- 2.2) estadísticas específicas por tipo (NO duplicar) ---
        last_vals = np.zeros(F, dtype=float)
        n_changes = np.zeros(F, dtype=float)
        skew_vals = np.zeros(F, dtype=float)
        kurt_vals = np.zeros(F, dtype=float)

        for j, (ch, ctype) in enumerate(zip(sensor_cols, channel_types)):
            col = seg[:, j]
            valid = ~np.isnan(col)
            v = col[valid]
            if v.size == 0:
                continue

            if _is_binary_type(ctype):
                last_vals[j] = float(v[-1])
                n_changes[j] = float(np.sum(v[1:] != v[:-1])) if v.size >= 2 else 0.0

            elif _is_wear_type(ctype):
                m = float(np.mean(v))
                sd = float(np.std(v, ddof=0))
                if sd > 1e-8:
                    centered = v - m
                    m3 = float(np.mean(centered ** 3))
                    m4 = float(np.mean(centered ** 4))
                    skew_vals[j] = m3 / (sd ** 3)
                    kurt_vals[j] = m4 / (sd ** 4)
                else:
                    skew_vals[j] = 0.0
                    kurt_vals[j] = 0.0

        # ✅ añadir UNA vez por parte (vectores tamaño F)
        all_feats.append(last_vals)
        all_names.extend([f"#p{p}@last value@{ch}" for ch in sensor_cols])

        all_feats.append(n_changes)
        all_names.extend([f"#p{p}@number changes@{ch}" for ch in sensor_cols])

        all_feats.append(skew_vals)
        all_names.extend([f"#p{p}@asymmetry@{ch}" for ch in sensor_cols])

        all_feats.append(kurt_vals)
        all_names.extend([f"#p{p}@propped@{ch}" for ch in sensor_cols])


    # -------------------------------------------------
    # 3) features de grupos deslizantes de tamaño 2..n
    # -------------------------------------------------
    for g in extra_group_sizes:
        if g <= 1 or g > n_parts:
            continue

        for start_idx in range(0, n_parts - g + 1):
            end_idx = start_idx + g
            seg_group = np.concatenate(parts[start_idx:end_idx], axis=0)  # (sum L, F)

            stats_this_group = stats_from_segment_nanaware(seg_group)

            group_label = "p" + "#".join(str(i) for i in range(start_idx, end_idx))

            # --- 3.1) estadísticas base ---
            for stat_name, stat_values in zip(stats_names, stats_this_group):
                all_feats.append(stat_values)
                for ch in sensor_cols:
                    all_names.append(f"#{group_label}@{stat_name}@{ch}")

            # --- 3.2) estadísticas específicas por tipo ---
            # --- 3.2) estadísticas específicas por tipo ---
            last_vals   = np.zeros(F, dtype=float)
            n_changes   = np.zeros(F, dtype=float)
            skew_vals   = np.zeros(F, dtype=float)
            kurt_vals   = np.zeros(F, dtype=float)

            for j, (ch, ctype) in enumerate(zip(sensor_cols, channel_types)):
                col = seg_group[:, j]
                valid = ~np.isnan(col)
                v = col[valid]

                if v.size == 0:
                    continue

                if _is_binary_type(ctype):
                    last_vals[j] = float(v[-1])
                    n_changes[j] = float(np.sum(v[1:] != v[:-1])) if v.size >= 2 else 0.0

                elif _is_wear_type(ctype):
                    m = float(np.mean(v))
                    std = float(np.std(v, ddof=0))
                    if std > 1e-8:
                        centered = v - m
                        m3 = float(np.mean(centered ** 3))
                        m4 = float(np.mean(centered ** 4))
                        skew_vals[j] = m3 / (std ** 3)
                        kurt_vals[j] = m4 / (std ** 4)
                    else:
                        skew_vals[j] = 0.0
                        kurt_vals[j] = 0.0

            # ✅ añadir UNA vez por grupo (vectores tamaño F)
            all_feats.append(last_vals)
            all_names.extend([f"#{group_label}@last value@{ch}" for ch in sensor_cols])

            all_feats.append(n_changes)
            all_names.extend([f"#{group_label}@number changes@{ch}" for ch in sensor_cols])

            all_feats.append(skew_vals)
            all_names.extend([f"#{group_label}@asymmetry@{ch}" for ch in sensor_cols])

            all_feats.append(kurt_vals)
            all_names.extend([f"#{group_label}@propped@{ch}" for ch in sensor_cols])




    feats_flat = np.concatenate(all_feats, axis=0)
    return feats_flat, all_names




if STATS:
    train_features = []
    test_features = []
    feature_names_stats = None

    for seg in all_train_segments:
        feats, names = extract_subwindow_features(
            seg,
            sensor_cols,
            channel_types=channel_types,
            n_parts=SW,
            void_descriptions_out=None,
        )
        train_features.append(feats)
        if feature_names_stats is None:
            feature_names_stats = names

    for seg in all_test_segments:
        feats, _ = extract_subwindow_features(
            seg,
            sensor_cols,
            channel_types=channel_types,
            n_parts=SW,
            void_descriptions_out=None,
        )
        test_features.append(feats)

    X_train2 = np.vstack(train_features).astype(np.float32)
    X_test2  = np.vstack(test_features).astype(np.float32)

    if USE_QUANTIZE2:
        X_train2 = quantize2(X_train2, Q_STEP2)
        X_test2  = quantize2(X_test2,  Q_STEP2)
        
    print("[stats]", X_train2.shape, X_test2.shape)        
else:
    X_train2 = None
    X_test2 = None
    feature_names_stats = None



if STATS and LOGIC:
    # unir lógica difusa (ya cuantizada si USE_QUANTIZE) + stats
    X_train = np.hstack([X_logic_train, X_train2])
    X_test  = np.hstack([X_logic_test,  X_test2])

    feature_names_fuzzy = train_feat_names_q
    feature_names = feature_names_fuzzy + feature_names_stats
    print("[fusion] Usando STATS+LOGIC: combinación completa")

elif STATS and not LOGIC:
    # solo estadísticas
    X_train = X_train2
    X_test  = X_test2

    feature_names = feature_names_stats
    print("[fusion] Usando solo STATS")

elif LOGIC and not STATS:
    # solo fuzzy/lógico (ya cuantizado si USE_QUANTIZE)
    X_train = X_logic_train
    X_test  = X_logic_test

    feature_names = train_feat_names_q
    print("[fusion] Usando solo LOGIC")

else:
    raise ValueError("Debes activar al menos uno de los flags STATS o LOGIC")


print("shapes:", X_train.shape, len(feature_names))
print("X_train_combined:", X_train.shape)
print("X_test_combined:",  X_test.shape)

# ======================================
# Guardar FLAT (vector exacto de train/test)
# ======================================
from pathlib import Path
import numpy as np

OUT_DIR = globals().get("FUZZY_DATA_OUT", "data/out")

np.savez_compressed(
    str(Path(OUT_DIR) / "features_train_flat.npz"),
    X=X_train.astype(np.float32),
    y=y_train.astype(np.int32),
    feature_names=np.array(feature_names, dtype=object),
    meta_json=np.array(json.dumps({
        "format": "flat_v1",
        "split": "train",
        "T5": int(T5),
        "SW": int(SW),
        "Q_STEP": float(Q_STEP),
        "Q_STEP2": float(Q_STEP2),
        "STATS": bool(STATS),
        "LOGIC": bool(LOGIC),
        "sensor_cols": list(sensor_cols),
        "channel_types": [str(t) for t in channel_types],
        "id2label": {int(k): str(v) for k,v in id2label.items()},
    }, ensure_ascii=False))
)

np.savez_compressed(
    str(Path(OUT_DIR) / "features_test_flat.npz"),
    X=X_test.astype(np.float32),
    y=y_test.astype(np.int32),
    feature_names=np.array(feature_names, dtype=object),
    meta_json=np.array(json.dumps({
        "format": "flat_v1",
        "split": "test",
        "T5": int(T5),
        "SW": int(SW),
        "Q_STEP": float(Q_STEP),
        "Q_STEP2": float(Q_STEP2),
        "STATS": bool(STATS),
        "LOGIC": bool(LOGIC),
        "sensor_cols": list(sensor_cols),
        "channel_types": [str(t) for t in channel_types],
        "id2label": {int(k): str(v) for k,v in id2label.items()},
    }, ensure_ascii=False))
)

print("[flat] Saved features_train_flat.npz and features_test_flat.npz in", OUT_DIR)

    
def check_nan_inf(X, feature_names, split_name="train"):
    X = np.asarray(X)

    print(f"\n=== Revisando {split_name} ===")
    any_nan = False

    for i, fname in enumerate(feature_names):
        col = X[:, i]

        if not np.isfinite(col).all():   # detecta NaN o Inf
            any_nan = True
            n_nan = np.isnan(col).sum()
            n_inf = np.isinf(col).sum()

            print(f"⚠️ Feature '{fname}' (col {i}) contiene:")
            print(f"    NaN: {n_nan}")
            print(f"    +Inf: {(col == np.inf).sum()}")
            print(f"    -Inf: {(col == -np.inf).sum()}")
            exit(0)

    if not any_nan:
        print("✓ No hay NaN ni Inf en este conjunto.")


rng = np.random.default_rng(123)

# elegir sample random
i = rng.integers(X_train.shape[0])

vals = X_train[i]

vals_str = ", ".join(f"{v:.2f}" for v in vals)

print(f"=== Sample {i} | label={id2label[y_train[i]]} ===")
print(f"=== Features fuzzy+stats (todos los valores) ===")
print(f"[{vals_str}]")

# Úsalo así:
check_nan_inf(X_train, feature_names, "train")
check_nan_inf(X_test,  feature_names, "test")

# =========================
# Baselines clásicos: KNN y SVM
# =========================

def _print_baseline_metrics(name, y_tr_pred, y_te_pred):
    """Imprime accuracy y F1 (macro y weighted) para train y test."""
    acc_tr = accuracy_score(y_train, y_tr_pred)
    f1_tr_macro = f1_score(y_train, y_tr_pred, average="macro")
    f1_tr_weighted = f1_score(y_train, y_tr_pred, average="weighted")

    acc_te = accuracy_score(y_test, y_te_pred)
    f1_te_macro = f1_score(y_test, y_te_pred, average="macro")
    f1_te_weighted = f1_score(y_test, y_te_pred, average="weighted")

    print(f"[{name}] "
          f"train_acc={acc_tr:.4f} | train_f1_macro={f1_tr_macro:.4f} | "
          f"train_f1_weighted={f1_tr_weighted:.4f} | "
          f"test_acc={acc_te:.4f} | test_f1_macro={f1_te_macro:.4f} | "
          f"test_f1_weighted={f1_te_weighted:.4f}")


print("\n==================== BASELINES: KNN y SVM ====================\n")

# --- KNN ---
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    n_jobs=-1,
)

knn.fit(X_train, y_train)
y_knn_tr = knn.predict(X_train)
y_knn_te = knn.predict(X_test)

_print_baseline_metrics("KNN", y_knn_tr, y_knn_te)


from sklearn.svm import LinearSVC
import numpy as np

svm = LinearSVC(C=1.0)
svm.fit(X_train, y_train)

y_svm_tr = svm.predict(X_train)
y_svm_te = svm.predict(X_test)

_print_baseline_metrics("SVM", y_svm_tr, y_svm_te)

# márgenes (N,C)
svm_margin_train = svm.decision_function(X_train)
svm_margin_test  = svm.decision_function(X_test)

def margin_to_proba(margin_row, temperature=2.0):
    z = np.asarray(margin_row, dtype=float) / temperature
    z = z - np.max(z)
    expz = np.exp(z)
    return expz / expz.sum()

svm_proba_train = np.vstack([margin_to_proba(m) for m in svm_margin_train])
svm_proba_test  = np.vstack([margin_to_proba(m) for m in svm_margin_test])


    
# =========================
# Entrenamiento con ExtraTrees (bosque interpretativo)
# =========================
clf = ExtraTreesClassifier(
    n_estimators=20,         # nº de árboles (más = más potencia)
    criterion="entropy",     # como C4.5
    max_depth=10,            # controla interpretabilidad
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_tree_tr = clf.predict(X_train)
y_tree_te = y_pred
_print_baseline_metrics("Tree", y_tree_tr, y_tree_te)

# =========================
# Evaluación
# =========================
print(f"Accuracy (cuantizado 0.1): {accuracy_score(y_test, y_pred):.4f}")
print("\nMatriz de confusión:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# Extracción de reglas más representativas
# =========================
print("\n=== Reglas de los primeros 3 árboles (depth <= 10) ===")
for i, tree in enumerate(clf.estimators_[:3]):
    rules = export_text(tree, feature_names=list(feature_names), decimals=1)  # 👈 muestra 1 decimal
    print(f"\n--- Árbol {i+1} ---")
    print(rules[:10])  # corta si es muy largo

# ==== Cell 22 ====
import numpy as np
import re

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




num_classes = len(np.unique(y_train))

# =========================
# Cuantización (opcional)
# =========================


from collections import Counter

counts = Counter(y_train.tolist())
class_weight = {c: len(y_train) / (len(counts) * cnt)
                for c, cnt in counts.items()}

weights = np.array([class_weight[c] for c in y_train])
print(counts)

# =========================
# Modelo XGBoost (multiclase)
# =========================
xgb = XGBClassifier(

    objective="multi:softprob",
    num_class=num_classes,
    tree_method="hist",     # usa "gpu_hist" si tienes GPU
    eval_metric="mlogloss",
    n_estimators=NTrees,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1
)

# Entrenamiento (sin early stopping para máxima compatibilidad)
xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False, sample_weight=weights)


xgb_proba_train = xgb.predict_proba(X_train)
xgb_proba_test  = xgb.predict_proba(X_test)

# =========================
# Evaluación
# =========================
y_pred = xgb.predict(X_test)
y_xgb_tr = xgb.predict(X_train)
y_xgb_te = y_pred
_print_baseline_metrics("XGBOOST", y_xgb_tr, y_xgb_te)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(pd.DataFrame(cm))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# =========================
# Importancias de características (gain)
# =========================
booster = xgb.get_booster()
raw_importances = booster.get_score(importance_type="gain")   # {'f123': gain}

rows = []
for fkey, gain in raw_importances.items():
    if fkey.startswith("f"):
        try:
            idx = int(fkey[1:])
            if 0 <= idx < len(feature_names):
                human_name = feature_names[idx]
            else:
                human_name = fkey
        except:
            human_name = fkey
    else:
        human_name = fkey

    rows.append({
        "xgb_feature": fkey,
        "feature": human_name,
        "gain": gain,
    })


def enrich_df_gain(df_gain: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Añade a df_gain:
      - feat_idx
      - feat_name
      - ftype: 'logic' / 'stats' / 'other'
      - sensor
      - sensor_type: 'location' / 'imu' / 'binary' / None
    """
    df = deepcopy(df_gain).reset_index(drop=True)

    # fXXXX -> índice de feature
    df["feat_idx"] = df["xgb_feature"].str.extract(r"f(\d+)").astype(int)

    # nombre legible
    df["feat_name"] = df["feat_idx"].apply(lambda i: feature_names[i])

    # tipo de feature por prefijo
    def _ftype(name: str) -> str:
        if name.startswith("#"):
            return "stats"
        if name.startswith("["):
            return "logic"
        return "other"

    df["ftype"] = df["feat_name"].apply(_ftype)

    # extraer sensor del nombre
    def _sensor_from_name(name: str) -> str | None:
        if name.startswith("#"):
            # "#p2#3#4@mean@magnetometer_x"
            try:
                body = name[1:]
                group_label, stat_name, sensor = body.split("@")
                return sensor
            except Exception:
                return None
        if name.startswith("["):
            # "[None] DINING_ROOM [Disabled] [in the middle]"
            parts = name.split()
            return parts[1] if len(parts) > 1 else None
        return None

    df["sensor"] = df["feat_name"].apply(_sensor_from_name)

    # tipo de sensor
    # tipo de sensor usando el mapa global ch2type
    def _sensor_type_safe(s):
        if s is None:
            return None
        # ch2type viene de: ch2type = dict(zip(sensor_cols, channel_types))
        return ch2type.get(s, None)


    df["sensor_type"] = df["sensor"].apply(_sensor_type_safe)

    # nos quedamos solo con las que tienen sensor reconocido
    df = df[df["sensor_type"].notna()].reset_index(drop=True)

    return df
    
df_gain = pd.DataFrame(rows).sort_values("gain", ascending=False)
df_gain_enriched = enrich_df_gain(df_gain, feature_names)

print("[pruning] df_gain_enriched shape:", df_gain_enriched.shape)
print(df_gain_enriched.head(5))

print("\n================ TOP 20 FEATURES por gain ================")
print(df_gain.head(20))

ix=0
#for idx, row in df_gain.iterrows():
    #print(f"Index: {idx}")
    #print(f"  xgb_feature: {row['xgb_feature']}")
    #print(f"  feature:     {row['feature']}")
    #print(f"  gain:        {row['gain']}")
    #print("-------------------------------------------------")
    #if ix> 150:
    #    break
    #ix=ix+1
print("==========================================================\n")


import numpy as np

def confidence_label(score: float) -> str:
    """
    Mapea score ∈ [0,1] a etiquetas lingüísticas de confianza.
    """
    if score < 0.10:
        return "very low confidence"
    elif score < 0.30:
        return "low confidence"
    elif score < 0.55:
        return "medium confidence"
    elif score < 0.80:
        return "high confidence"
    else:
        return "very high confidence"


def pick_label_from_proba(proba_row, train_mode, rng):
    p = np.asarray(proba_row, dtype=float)
    p = np.clip(p, 0.0, None)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)

    if train_mode:
        cid = int(rng.choice(len(p), p=p))   # 🎲 muestreo
    else:
        cid = int(np.argmax(p))              # 🏆 mejor clase

    return cid, float(p[cid])


def pick_stats_random_weighted(df_stats, row_vals, alpha=2.0):
    """
    Selecciona una feature estadística de forma aleatoria ponderada por:
        score = (gain * |valor|)^alpha

    df_stats: subset de df_gain_enriched ya filtrado por:
              - sensor concreto
              - ftype == 'stats'
    row_vals: vector (n_features,) con los valores del sample
    alpha:    >1 afila (más cerca del top), =1 proporcional, <1 más exploración

    Devuelve:
        (feat_name, value) o None si df_stats está vacío
    """
    if df_stats.empty:
        return None

    # índices de feature y valores de este sample
    feat_idxs = df_stats["feat_idx"].to_numpy()
    vals = row_vals[feat_idxs].astype(float)         # valores stats (z-score o lo que sea)

    gains = df_stats["gain"].astype(float).to_numpy()
    gains = np.clip(gains, 0.0, None)

    # score = gain * |valor| → anomalías (grandes en módulo) con alto gain
    scores = gains * 1
    scores = np.clip(scores, 0.0, None)

    if np.all(scores == 0):
        # fallback: distribución uniforme
        probs = np.ones_like(scores) / len(scores)
    else:
        scores = scores ** alpha
        probs = scores / scores.sum()

    choice = np.random.choice(len(df_stats), p=probs)
    row = df_stats.iloc[int(choice)]

    feat_name = row["feat_name"]
    value = float(vals[int(choice)])

    return feat_name, value


    

def format_stats_feature(feat_name: str, value: float) -> str:
    """
    feat_name ej: "#p2#3#4@mean@magnetometer_x"
                  "#p1@void@HALL"
    """
    try:
        assert feat_name.startswith("#")
        body = feat_name[1:]
        group_label, stat_name, sensor = body.split("@")
    except Exception:
        return f"{feat_name} is {value:.2f}"

    # group_label: "p2#3#4" o "p3"
    label_wo_p = group_label[1:]          # "2#3#4"
    parts = label_wo_p.split("#")         # ["2","3","4"] o ["3"]
    idxs = [int(p) for p in parts]

    if len(idxs) == 1:
        span = f"[{idxs[0]}s]"
    else:
        span = f"[{idxs[0]}s,{idxs[-1]}s]"

    return f"{stat_name} of {sensor} between {span} is {value:.2f}"
    
    
def sensor_all_minus_one_raw(raw_window, raw_channels, sensor):
    """
    Devuelve True si en la ventana cruda del sensor todos los valores son -1.
    """
    if sensor not in raw_channels:
        return True
    
    # Asegurar que raw_window es np.ndarray (T, F)
    arr = np.asarray(raw_window, dtype=float)
    
    # Si está transpuesta (F, T), la giramos
    if arr.shape[1] == len(raw_channels):
        pass  # (T, F) → OK
    elif arr.shape[0] == len(raw_channels):
        arr = arr.T  # (F, T) → (T, F)
    else:
        raise ValueError(
            f"raw_window shape {arr.shape} no corresponde con raw_channels len={len(raw_channels)}"
        )
    
    col = raw_channels.index(sensor)
    vals = arr[:, col]

    return np.all(vals == 0.0)

def format_logic_feature(feat_name: str, value: float) -> str:
    """
    Limpia feat_name eliminando corchetes y pasando todo a minúsculas.
    Ejemplo entrada:
      "[None] loc_toileting [is Low] [toward the end]"
    Salida:
      "none loc_toileting is low toward the end (1.00)"
    """
    try:
        # quitamos corchetes
        cleaned = feat_name.replace("[", "").replace("]", "")
        # pasamos todo a minúsculas
        cleaned = cleaned.lower().strip()
        # formateamos
        return f"{cleaned} ({value:.2f})"
    except Exception:
        return f"{feat_name} ({value:.2f})"
    
def pick_best_for_sensor(
    df_gain_enriched: pd.DataFrame,
    row_vals: np.ndarray,
    sensor_name: str,
    mode: str,                    # "logic" o "stats"
    use_membership_for_logic: bool = True,
    alpha_stats: float = 2.0,
):
    """
    Devuelve texto formateado (o None) para un sensor dado y un modo.

    - mode == "logic": elige la feature lógica con mayor score = gain * membership (si se pide)
    - mode == "stats": elige una feature estadística de forma aleatoria ponderada
                       por (gain * |valor|)^alpha_stats
    """
    assert mode in ("logic", "stats")

    # subset de features para ese sensor y modo
    dft = df_gain_enriched[
        (df_gain_enriched["sensor"] == sensor_name)
        & (df_gain_enriched["ftype"] == mode)
    ]
    if dft.empty:
        return None

    # valores de la fila (membership o valor estadístico) para este sample
    feat_idxs = dft["feat_idx"].to_numpy()
    vals = row_vals[feat_idxs].astype(float)


    gains = dft["gain"].astype(float).to_numpy()

    if mode == "logic":
        # score = gain * membership (si se quiere) o solo gain
        if use_membership_for_logic:
            scores = gains * np.clip(vals, 0.0, None)
            if np.all(scores <= 0):
                scores = gains
        else:
            scores = gains

        best_idx = int(np.argmax(scores))
        best_row = dft.iloc[best_idx]
        best_name = best_row["feat_name"]
        best_val = float(vals[best_idx])

        return format_logic_feature(best_name, best_val)

    else:  # mode == "stats"
        # usamos selección aleatoria ponderada por gain * |valor|
        feat = pick_stats_random_weighted(dft, row_vals, alpha=alpha_stats)
        if feat is not None:
            best_name, best_val = feat
        else:
            # fallback determinista si algo raro pasa
            scores = gains * np.abs(vals)
            if np.all(scores == 0):
                scores = gains
            best_idx = int(np.argmax(scores))
            best_row = dft.iloc[best_idx]
            best_name = best_row["feat_name"]
            best_val = float(vals[best_idx])

        return format_stats_feature(best_name, best_val)


def pick_best_for_sensor_k(
    df_gain_enriched: pd.DataFrame,
    row_vals: np.ndarray,
    sensor_name: str,
    mode: str,                    # "logic" o "stats"
    top_k: int = 1,
    use_membership_for_logic: bool = True,
    alpha_stats: float = 2.0,
):
    """
    Igual que pick_best_for_sensor, pero devuelve las K mejores features.

    Retorna:
        - lista de textos formateados (longitud K o menos si no hay tantas)
        - [] si no hay features disponibles
    """
    assert mode in ("logic", "stats")
    if top_k <= 0:
        return []

    # subset de features para ese sensor y tipo
    dft = df_gain_enriched[
        (df_gain_enriched["sensor"] == sensor_name)
        & (df_gain_enriched["ftype"] == mode)
    ]
    if dft.empty:
        return []

    feat_idxs = dft["feat_idx"].to_numpy()
    vals = row_vals[feat_idxs].astype(float)
    gains = dft["gain"].astype(float).to_numpy()

    # ----------------------------------------------------------------------
    # LOGICAL FEATURES
    # ----------------------------------------------------------------------
    if mode == "logic":
        if use_membership_for_logic:
            scores = gains * np.clip(vals, 0.0, None)
            if np.all(scores <= 0):
                scores = gains
        else:
            scores = gains

        # top-k ordenados por score
        order = np.argsort(scores)[::-1][:top_k]

        out_list = []
        for idx in order:
            row = dft.iloc[idx]
            f_name = row["feat_name"]
            f_val  = float(vals[idx])
            out_list.append(format_logic_feature(f_name, f_val))

        return out_list

    # ----------------------------------------------------------------------
    # STATISTICAL FEATURES
    # ----------------------------------------------------------------------
    else:
        # score base
        scores = gains * np.abs(vals)
        scores = np.clip(scores, 0.0, None)

        if np.all(scores == 0):
            order = np.arange(len(scores))
        else:
            scores_pow = scores ** alpha_stats
            order = np.argsort(scores_pow)[::-1]

        order = order[:top_k]

        out_list = []
        for idx in order:
            row = dft.iloc[idx]
            f_name = row["feat_name"]
            f_val  = float(vals[idx])
            out_list.append(format_stats_feature(f_name, f_val))

        return out_list

import re

def _replace_channel_tokens(feat_list, ch, raw_channels):
    out = []
    token = f"<{ch}>" if ch in raw_channels else f"<unknown_{ch}>"

    # patrón seguro: solo reemplaza el canal separado por límites
    pattern = r'\b' + re.escape(ch) + r'\b'

    for t in feat_list:
        if t is None:
            out.append(None)
        else:
            out.append(re.sub(pattern, token, t))
    return out


def toon_record_for_sample(
    sample_idx,
    X_feats,
    df_gain_enriched,
    raw_window,
    raw_channels,
    use_membership_for_logic=True,
    top_k_logic=8,
    top_k_stats=8,
):
    row_vals = X_feats[sample_idx]

    W = {}
    B = {}


    # LOCATION
    for s in WEAR_SENSORS:

        if sensor_all_minus_one_raw(raw_window, raw_channels, s):
            W[s] = [f"<{s}>" + " is None"]
            W[s] = [f""]
            continue

        logic_list = pick_best_for_sensor_k(
            df_gain_enriched, row_vals, s, "logic",
            top_k=top_k_logic,
            use_membership_for_logic=use_membership_for_logic
        )

        stats_list = pick_best_for_sensor_k(
            df_gain_enriched, row_vals, s, "stats",
            top_k=top_k_stats,
        )

        feats = logic_list + stats_list
        W[s] = _replace_channel_tokens(feats, s, raw_channels)

    # BINARY
    for s in BINARY_SENSORS:

        if sensor_all_minus_one_raw(raw_window, raw_channels, s):
            B[s] = [f"<{s}>" + " is None"]
            B[s] = [f""]
            continue

        logic_list = pick_best_for_sensor_k(
            df_gain_enriched, row_vals, s, "logic",
            top_k=top_k_logic,
            use_membership_for_logic=use_membership_for_logic
        )

        stats_list = pick_best_for_sensor_k(
            df_gain_enriched, row_vals, s, "stats",
            top_k=top_k_stats,
        )

        feats = logic_list + stats_list
        B[s] = _replace_channel_tokens(feats, s, raw_channels)

    return {"W": W, "B": B}




import json
from pathlib import Path




def build_and_save_toon_descriptions(
    X_feats: np.ndarray,
    feature_names: list[str],
    df_gain: pd.DataFrame,
    out_path: str | Path,
    raw_records,
    split: str,   # 👈 AÑADIDO
    use_membership_for_logic: bool = True,
    svm_proba: np.ndarray | None = None,
    xgb_proba: np.ndarray | None = None,
    id2label: dict[int, str] | None = None,
    seed: int = 1234,
):

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_gain_en = enrich_df_gain(df_gain, feature_names)

    n_samples = X_feats.shape[0]
    print(f"[build_and_save_toon_descriptions] {n_samples} samples -> {out_path}")

    train_mode = (split.lower() == "train")
    rng = np.random.default_rng(seed)

    with out_path.open("w", encoding="utf-8") as f_out:
        for i in range(n_samples):
            rec = toon_record_for_sample(
                sample_idx=i,
                X_feats=X_feats,
                df_gain_enriched=df_gain_en,
                raw_window=raw_records[i]["window"],
                raw_channels=raw_records[i]["channels"],
                use_membership_for_logic=use_membership_for_logic,
            )

            # -----------------------------
            # 👇 EXPERTS + RAG DESCRIPTION
            # -----------------------------
            if (svm_proba is not None) and (xgb_proba is not None) and (id2label is not None):
                svm_c, svm_s = pick_label_from_proba(
                    svm_proba[i], train_mode=train_mode, rng=rng
                )
                xgb_c, xgb_s = pick_label_from_proba(
                    xgb_proba[i], train_mode=train_mode, rng=rng
                )

                svm_lbl = id2label[int(svm_c)]
                xgb_lbl = id2label[int(xgb_c)]

                svm_conf = confidence_label(svm_s)
                xgb_conf = confidence_label(xgb_s)

                rec["rag_description"] = (
                    f"Expert SVM says answer is <{svm_lbl}> "
                    f"({svm_s*100:.0f}%, {svm_conf}). "
                    f"Expert XGBoost answer is  <{xgb_lbl}> "
                    f"({xgb_s*100:.0f}%, {xgb_conf})."
                )

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[build_and_save_toon_descriptions] DONE")



import json
from pathlib import Path





    
# tras entrenar SVM y XGB y tener svm_proba_* / xgb_proba_*
build_and_save_toon_descriptions(
    X_feats=X_train,
    feature_names=feature_names,
    df_gain=df_gain,
    out_path=globals()["FUZZY_DATA_OUT"] + "/descriptions_train_toon.jsonl",
    raw_records=train_records_kept,
    split="train",
    svm_proba=svm_proba_train,
    xgb_proba=xgb_proba_train,
    id2label=id2label,
    seed=1234,
)

build_and_save_toon_descriptions(
    X_feats=X_test,
    feature_names=feature_names,
    df_gain=df_gain,
    out_path=globals()["FUZZY_DATA_OUT"] + "/descriptions_test_toon.jsonl",
    raw_records=test_records_kept,
    split="test",
    svm_proba=svm_proba_test,
    xgb_proba=xgb_proba_test,
    id2label=id2label,
    seed=1234,
)



import numpy as np
import json
from pathlib import Path

def pack_logic_to_u8(X_logic: np.ndarray, step: float) -> np.ndarray:
    # X_logic en [0,1] (ya cuantizado). step>0
    X = np.asarray(X_logic, dtype=np.float32)
    X = np.clip(X, 0.0, 1.0)
    q = np.rint(X / float(step)).astype(np.int32)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q

import json
import numpy as np
from pathlib import Path

def save_features_npz_by_type(
    out_path: str,
    X_wear: np.ndarray,   # (N, Cw, d_w) o None
    X_bin:  np.ndarray,   # (N, Cb, d_b) o None
    y: np.ndarray,
    wear_cols: list[str],
    bin_cols: list[str],
    logic_step: float | None = None,
    stats_step: float | None = None,
    extra_meta: dict | None = None,
):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    y_i32 = np.asarray(y, dtype=np.int32)

    X_wear_f16 = None if X_wear is None else np.asarray(X_wear, dtype=np.float16)
    X_bin_f16  = None if X_bin  is None else np.asarray(X_bin,  dtype=np.float16)

    meta = {
        "format": "by_type_v1",

        "wear_cols": list(wear_cols),
        "bin_cols": list(bin_cols),

        "wear_feat_dim": int(X_wear.shape[-1]) if X_wear is not None else None,
        "bin_feat_dim":  int(X_bin.shape[-1])  if X_bin  is not None else None,
        "logic_step": float(logic_step) if logic_step is not None else None,
        "stats_step": float(stats_step) if stats_step is not None else None,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_json = json.dumps(meta, ensure_ascii=False)

    np.savez_compressed(
        out_path,
        X_wear_f16=X_wear_f16,
        X_bin_f16=X_bin_f16,
        y=y_i32,
        meta_json=np.array(meta_json),
    )
    print(f"[save_features_npz_by_type] Saved -> {out_path}")
    print("  X_wear_f16:", None if X_wear_f16 is None else X_wear_f16.shape)
    print("  X_bin_f16: ", None if X_bin_f16  is None else X_bin_f16.shape)
    print("  y:         ", y_i32.shape)


def split_logic_by_type(X_logic, sensor_cols, channel_types, T5, Q):
    d_w = 5 * T5 * Q
    d_b = 2 * T5 * Q

    wear_cols = [c for c,t in zip(sensor_cols, channel_types) if str(t).upper()=="WEAR"]
    bin_cols  = [c for c,t in zip(sensor_cols, channel_types) if str(t).upper()=="BINARY"]

    N = X_logic.shape[0]
    Xw = np.zeros((N, len(wear_cols), d_w), dtype=np.float32)
    Xb = np.zeros((N, len(bin_cols),  d_b), dtype=np.float32)

    pos = 0; iw = 0; ib = 0
    for ch,tp in zip(sensor_cols, channel_types):
        if str(tp).upper() == "WEAR":
            Xw[:, iw, :] = X_logic[:, pos:pos+d_w]
            pos += d_w; iw += 1
        else:
            Xb[:, ib, :] = X_logic[:, pos:pos+d_b]
            pos += d_b; ib += 1

    assert pos == X_logic.shape[1]
    return Xw, Xb, wear_cols, bin_cols


def split_stats_by_type(X_stats, sensor_cols, channel_types):
    # stats: mismo dS por canal en tu extractor (si no, dilo)
    C = len(sensor_cols)
    assert X_stats.shape[1] % C == 0
    dS = X_stats.shape[1] // C

    wear_cols = [c for c,t in zip(sensor_cols, channel_types) if str(t).upper()=="WEAR"]
    bin_cols  = [c for c,t in zip(sensor_cols, channel_types) if str(t).upper()=="BINARY"]

    N = X_stats.shape[0]
    Xw = np.zeros((N, len(wear_cols), dS), dtype=np.float32)
    Xb = np.zeros((N, len(bin_cols),  dS), dtype=np.float32)

    pos = 0; iw = 0; ib = 0
    for ch,tp in zip(sensor_cols, channel_types):
        if str(tp).upper() == "WEAR":
            Xw[:, iw, :] = X_stats[:, pos:pos+dS]
            iw += 1
        else:
            Xb[:, ib, :] = X_stats[:, pos:pos+dS]
            ib += 1
        pos += dS

    assert pos == X_stats.shape[1]
    return Xw, Xb, wear_cols, bin_cols

import numpy as np
import pandas as pd

def _topk_local_by_gain(df_sub: pd.DataFrame, local_idx_col: str, k: int) -> list[int]:
    """
    Selecciona top-k por gain y devuelve índices locales (dentro del canal).
    Si faltan features en df_sub, simplemente no entran -> se consideran gain=0.
    """
    if k <= 0:
        return []
    if df_sub is None or len(df_sub) == 0:
        return []  # si no hay info, no elegimos nada (quedará todo a 0 si no haces fallback)
    df2 = df_sub.sort_values("gain", ascending=False).head(int(k))
    return df2[local_idx_col].astype(int).tolist()


def prune_by_gain_enriched_keep50(
    df_gain_enriched: pd.DataFrame,
    sensor_cols: list[str],
    channel_types: list[str],
    X_logic_train_2d: np.ndarray,   # (N, D_logic_total)
    X_stats_train_2d: np.ndarray,   # (N, D_stats_total)
    T5: int,
    Q: int,
    dS: int,                        # stats por canal (constante)
    keep_ratio: float = 0.50,
):
    """
    Devuelve tensores reducidos por tipo:
      X_wear_red: (N, Cw, dLw_keep + dS_keep)
      X_bin_red : (N, Cb, dLb_keep + dS_keep)
    y un meta con los índices usados por canal.

    IMPORTANT:
    - asume que los bloques están en el ORDEN sensor_cols
    - asume dS constante por canal
    - logic wear = 5*T5*Q, logic bin = 2*T5*Q
    """
    keep_ratio = float(keep_ratio)
    assert 0 < keep_ratio <= 1.0

    dL_wear = int(5 * T5 * Q)
    dL_bin  = int(2 * T5 * Q)

    # offsets globales en X_logic según sensor_cols y tipo
    logic_offsets = {}
    pos = 0
    for ch, tp in zip(sensor_cols, channel_types):
        tpU = str(tp).upper()
        dL = dL_wear if tpU == "WEAR" else dL_bin
        logic_offsets[ch] = (pos, dL)
        pos += dL
    D_logic_total = pos

    # offsets stats globales: dS por canal en el MISMO orden sensor_cols
    stats_offsets = {ch: (i * dS, dS) for i, ch in enumerate(sensor_cols)}
    D_stats_total = len(sensor_cols) * dS

    # sanity
    assert X_logic_train_2d.shape[1] == D_logic_total
    assert X_stats_train_2d.shape[1] == D_stats_total

    # canales por tipo
    wear_cols = [c for c, t in zip(sensor_cols, channel_types) if str(t).upper() == "WEAR"]
    bin_cols  = [c for c, t in zip(sensor_cols, channel_types) if str(t).upper() != "WEAR"]

    # tamaños keep (mismos para todos los canales de ese tipo)
    kL_wear = int(np.ceil(dL_wear * keep_ratio))
    kL_bin  = int(np.ceil(dL_bin  * keep_ratio))
    kS      = int(np.ceil(dS      * keep_ratio))

    # salidas
    N = X_logic_train_2d.shape[0]
    X_wear_red = None
    X_bin_red  = None

    if len(wear_cols) > 0:
        X_wear_red = np.zeros((N, len(wear_cols), kL_wear + kS), dtype=np.float32)
    if len(bin_cols) > 0:
        X_bin_red  = np.zeros((N, len(bin_cols),  kL_bin  + kS), dtype=np.float32)

    # meta trazable
    prune_meta = {
        "method": "xgb_gain_keep_ratio_per_channel_pack",
        "keep_ratio": keep_ratio,
        "kS": kS,
        "kL_wear": kL_wear,
        "kL_bin": kL_bin,
        "per_channel": {}
    }

    # construimos por canal (pack)
    wear_i = 0
    bin_i  = 0

    for ch, tp in zip(sensor_cols, channel_types):
        tpU = str(tp).upper()
        lo0, dL = logic_offsets[ch]
        so0, _  = stats_offsets[ch]

        # --- subset df_gain_enriched de este canal ---
        # df_gain_enriched viene de enrich_df_gain(...): tiene feat_idx, ftype, sensor, gain, etc. :contentReference[oaicite:1]{index=1}
        df_ch_logic = df_gain_enriched[(df_gain_enriched["sensor"] == ch) & (df_gain_enriched["ftype"] == "logic")].copy()
        df_ch_stats = df_gain_enriched[(df_gain_enriched["sensor"] == ch) & (df_gain_enriched["ftype"] == "stats")].copy()

        # convertimos feat_idx -> local dentro del canal
        # ojo: feat_idx para stats está en el vector "feature_names" combinado (logic+stats),
        # pero aquí trabajamos con X_stats_train_2d ya “separado”, así que necesitamos:
        #   local_stats = (feat_idx - D_logic_total) - so0
        df_ch_logic["local_logic"] = df_ch_logic["feat_idx"] - lo0
        df_ch_stats["local_stats"] = (df_ch_stats["feat_idx"] - D_logic_total) - so0

        # filtramos rangos válidos
        df_ch_logic = df_ch_logic[(df_ch_logic["local_logic"] >= 0) & (df_ch_logic["local_logic"] < dL)]
        df_ch_stats = df_ch_stats[(df_ch_stats["local_stats"] >= 0) & (df_ch_stats["local_stats"] < dS)]

        # top-k
        if tpU == "WEAR":
            keepL = _topk_local_by_gain(df_ch_logic, "local_logic", kL_wear)
        else:
            keepL = _topk_local_by_gain(df_ch_logic, "local_logic", kL_bin)
        keepS = _topk_local_by_gain(df_ch_stats, "local_stats", kS)

        # fallback suave: si por lo que sea no hay suficientes, rellenamos con los primeros índices
        # (así nunca dejas el canal vacío)
        if tpU == "WEAR":
            if len(keepL) < kL_wear:
                fill = [i for i in range(dL_wear) if i not in set(keepL)]
                keepL = keepL + fill[: (kL_wear - len(keepL))]
        else:
            if len(keepL) < kL_bin:
                fill = [i for i in range(dL_bin) if i not in set(keepL)]
                keepL = keepL + fill[: (kL_bin - len(keepL))]

        if len(keepS) < kS:
            fill = [i for i in range(dS) if i not in set(keepS)]
            keepS = keepS + fill[: (kS - len(keepS))]

        keepL = np.array(keepL, dtype=np.int32)
        keepS = np.array(keepS, dtype=np.int32)

        prune_meta["per_channel"][ch] = {
            "type": tpU,
            "keep_logic_local": keepL.tolist(),
            "keep_stats_local": keepS.tolist(),
        }

        # --- extraemos datos reales y empaquetamos ---
        X_logic_ch = X_logic_train_2d[:, lo0:lo0+dL]     # (N, dL)
        X_stats_ch = X_stats_train_2d[:, so0:so0+dS]     # (N, dS)

        if tpU == "WEAR" and X_wear_red is not None:
            X_wear_red[:, wear_i, :keepL.shape[0]] = X_logic_ch[:, keepL]
            X_wear_red[:, wear_i, keepL.shape[0]:] = X_stats_ch[:, keepS]
            wear_i += 1
        elif tpU != "WEAR" and X_bin_red is not None:
            X_bin_red[:, bin_i, :keepL.shape[0]] = X_logic_ch[:, keepL]
            X_bin_red[:, bin_i, keepL.shape[0]:] = X_stats_ch[:, keepS]
            bin_i += 1

    return X_wear_red, X_bin_red, wear_cols, bin_cols, prune_meta



X_wear_train_red, X_bin_train_red, wear_cols, bin_cols, prune_meta = prune_by_gain_enriched_keep50(
    df_gain_enriched=df_gain_enriched,
    sensor_cols=sensor_cols,
    channel_types=channel_types,
    X_logic_train_2d=X_logic_train,    # (N, D_logic_total)
    X_stats_train_2d=X_train2,    # (N, D_stats_total)
    T5=T5,
    Q=T5,
    dS=X_train2.shape[1] // len(sensor_cols),
    keep_ratio=0.8,
)

X_wear_test_red, X_bin_test_red, _, _, _ = prune_by_gain_enriched_keep50(
    df_gain_enriched=df_gain_enriched,
    sensor_cols=sensor_cols,
    channel_types=channel_types,
    X_logic_train_2d=X_logic_test,
    X_stats_train_2d=X_test2,
    T5=T5,
    Q=T5,
    dS=X_test2.shape[1] // len(sensor_cols),
    keep_ratio=0.8,
)

# Sustituye los tensores originales por los reducidos (los que se guardan)
X_wear_train = X_wear_train_red
X_wear_test  = X_wear_test_red
X_bin_train  = X_bin_train_red
X_bin_test   = X_bin_test_red

# -----------------------------
# CALLS (train / test)
# -----------------------------
OUT_DIR = globals().get("FUZZY_DATA_OUT", "data/out")
out_train = str(Path(OUT_DIR) / "features_train_zsym.npz")
out_test  = str(Path(OUT_DIR) / "features_test_zsym.npz")



save_features_npz_by_type(
    out_train,
    X_wear=X_wear_train,
    X_bin=X_bin_train,
    y=y_train,
    wear_cols=wear_cols,
    bin_cols=bin_cols,
    logic_step=Q_STEP,
    stats_step=Q_STEP2,
    extra_meta={
        "split":"train",
        "num_labels": int(num_labels),
        "id2label": {int(k): str(v) for k,v in id2label.items()},
        "T5": int(T5),
        "Q": int(T5),
        "SW": int(SW),
        "stats_group_sizes": [2,3],
        "d_logic_wear": int(5*T5*T5),
        "d_logic_bin": int(2*T5*T5),
    }
)

save_features_npz_by_type(
    out_test,
    X_wear=X_wear_test,
    X_bin=X_bin_test,
    y=y_test,
    wear_cols=wear_cols,
    bin_cols=bin_cols,
    logic_step=Q_STEP,
    stats_step=Q_STEP2,
    extra_meta={
        "split":"test",
        "num_labels": int(num_labels),
        "id2label": {int(k): str(v) for k,v in id2label.items()},
        "T5": int(T5),
        "Q": int(T5),
        "SW": int(SW),
        "stats_group_sizes": [2,3],
        "d_logic_wear": int(5*T5*T5),
        "d_logic_bin": int(2*T5*T5),
    }
)



from pathlib import Path
import json
import joblib

def save_bundle(out_dir: str,
                knn, svm, tree, xgb,
                feature_names, label2id, id2label,
                extra_meta: dict | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Modelos sklearn a joblib
    joblib.dump(knn,  out / "knn.joblib")
    joblib.dump(svm,  out / "svm_linear.joblib")
    joblib.dump(tree, out / "tree_extratrees.joblib")

    # 2) XGBoost: guardar modelo nativo (más portable que joblib)
    xgb.save_model(str(out / "xgb.json"))

    # 3) Metadata (todo lo necesario para reconstruir entradas/salidas)
    meta = {
        "feature_names": list(feature_names),
        "label2id": {str(k): int(v) for k, v in label2id.items()},
        "id2label": {str(int(k)): str(v) for k, v in id2label.items()},
    }
    if extra_meta:
        meta.update(extra_meta)

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[save_bundle] saved to: {out.resolve()}")

# Ejemplo de uso
save_bundle(
    out_dir=globals().get("FUZZY_DATA_OUT", "output") + "/models_bundle",
    knn=knn,
    svm=svm,
    tree=clf,
    xgb=xgb,
    feature_names=feature_names,
    label2id=label2id,
    id2label=id2label,
    extra_meta={
        "Q_STEP": float(Q_STEP),
        "Q_STEP2": float(Q_STEP2),
        "STATS": bool(STATS),
        "LOGIC": bool(LOGIC),
        "T5": int(T5),
        "SW": int(SW),
        "NTrees": int(NTrees),
        "sensor_cols": list(sensor_cols),
        "channel_types": [str(t) for t in channel_types],
    }
)


