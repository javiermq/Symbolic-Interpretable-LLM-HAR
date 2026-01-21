import yaml
import argparse, os
import json
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# --- Argumentos ---
ap = argparse.ArgumentParser(description="Run fuzzy_terms pipeline (converted from notebook)")
ap.add_argument("--out", dest="out_dir", default="output",
                help="Main output folder for figures, logs, etc.")
ap.add_argument("--FUZZY_DATA_OUT", dest="data_out_dir", default="data/mhealth",
                help="Folder for final dataset files (train2.jsonl, val2.jsonl)")
ap.add_argument("--train_jsonl", dest="train_jsonl", required=True,
                help="Ruta al jsonl de ventanas de train")
ap.add_argument("--test_jsonl", dest="test_jsonl", required=True,
                help="Ruta al jsonl de ventanas de test")
                
ap.add_argument("--sensor_cfg", dest="sensor_cfg", default=None,
                help="Ruta al YAML con la definicion de tipos de sensor (ej. marble.cfg)")

# ?? Nuevo Argumento para control de normalizacion
ap.add_argument("--normalize", action="store_true", default=False,
                help="Aplica normalizacion Z-SCORE condicional (por canal, usando stats de train).")
                

args = ap.parse_args()

SENSOR_CFG_PATH = args.sensor_cfg

globals()["FUZZY_OUT"] = args.out_dir
globals()["FUZZY_DATA_OUT"] = args.data_out_dir
globals()["FUZZY_TRAIN_JSONL"] = args.train_jsonl
globals()["FUZZY_TEST_JSONL"] = args.test_jsonl
# ?? Hacer visible la bandera de normalizacion
DO_NORMALIZATION = args.normalize

os.environ["FUZZY_OUT"] = args.out_dir


os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(args.data_out_dir, exist_ok=True)


print(f"[fuzzy_terms] Output dir:     {args.out_dir}")
print(f"[fuzzy_terms] Data out dir:   {args.data_out_dir}")
print(f"[fuzzy_terms] Train jsonl:    {args.train_jsonl}")
print(f"[fuzzy_terms] Test jsonl:     {args.test_jsonl}")
print(f"[fuzzy_terms] Normalizar:     {DO_NORMALIZATION}") # Muestra el estado

# --- Funciones Auxiliares (omitiendo _normalize_type_name) ---

def load_sensor_types_from_yaml(cfg_path: str, channel_names):
    """
    Lee un YAML de definicion de sensores.
    """
    if cfg_path is None:
        return {}

    with open(cfg_path, "r", encoding="utf-8") as f:
        # por si el cfg lleva tabs, los convertimos a espacios
        text = f.read().replace("\t", "  ")

    cfg = yaml.safe_load(text)
    sensors_section = cfg.get("sensors", {})

    channel_set = set(channel_names)
    type_map = {}

    for raw_type, lst in sensors_section.items():
        if not isinstance(lst, (list, tuple)):
            continue

        tname = raw_type.strip()

        for ch in lst:
            if isinstance(ch, str) and ch in channel_set:
                type_map[ch] = tname

    return type_map

# --- Lectura de Datos ---

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
# Convertimos a segmentos numericos
# =========================
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

        return arr, sensor_cols

    # --- Formato antiguo: dict {canal: [T]} ---
    if isinstance(win, dict):
        sensor_cols = list(win.keys())
        arr_cols = [np.asarray(win[c], dtype=float) for c in sensor_cols]
        arr = np.stack(arr_cols, axis=1)
        return arr, sensor_cols

    raise ValueError("Formato de 'window' no reconocido.")


# Detecta columnas y tamano temporal
example_arr, sensor_cols = window_to_array_generic(example_rec)
T = example_arr.shape[0]
WINDOW_SIZE_ROWS = T
print("[fuzzy] WINDOW_SIZE_ROWS:", WINDOW_SIZE_ROWS)
print("[fuzzy] sensor_cols:", sensor_cols)

sensor_type_map = load_sensor_types_from_yaml(SENSOR_CFG_PATH, sensor_cols)


# =========================
# Convertimos TODOS los registros a arrays (T, F)
# =========================
def _normalize_label(x):
    # Normaliza etiquetas a string "unknown" si vienen como None o vacias
    if not isinstance(x, str) or not x.strip():
        return "unknown"
    return x.strip()

all_train_segments = []
all_train_labels   = []

for i, rec in enumerate(train_records):
    seg, _ = window_to_array_generic(rec)

    raw_label = rec.get("label", None)
    lbl = _normalize_label(raw_label)

    if lbl == "unknown" or lbl == "TRANSITION":
        print("unkonw",{
            "index": i,
            "raw_label": raw_label,
            "scenario": rec.get("scenario"),
            "instance": rec.get("instance"),
            "subject": rec.get("subject"),
            "window_shape": np.asarray(rec["window"]).shape if "window" in rec else None,
        })
        continue
        

    all_train_segments.append(seg)
    all_train_labels.append({
        "label": lbl,
        "context": rec.get("context", "unknown"),
        "description": rec.get("description", "unknown"),
        "activity": rec.get("activity", lbl),
        "subject": rec.get("subject", -1),
        "activity_name": lbl,
        "segments": 1,
        "scenario": rec.get("scenario", None),
        "instance": rec.get("instance", None),
    })

all_test_segments = []
all_test_labels   = []
for rec in test_records:
    seg, _ = window_to_array_generic(rec)

    raw_label = rec.get("label", None)
    lbl = _normalize_label(raw_label)

    if lbl == "unknown" or lbl == "TRANSITION":
        print("unkonw",{
            "index": i,
            "raw_label": raw_label,
            "scenario": rec.get("scenario"),
            "instance": rec.get("instance"),
            "subject": rec.get("subject"),
            "window_shape": np.asarray(rec["window"]).shape if "window" in rec else None,
        })
        continue

    all_test_segments.append(seg)
    all_test_labels.append({
        "label": lbl,
        "context": rec.get("context", "unknown"),
        "description": rec.get("description", "unknown"),
        "activity": rec.get("activity", lbl),
        "subject": rec.get("subject", -1),
        "activity_name": lbl,
        "segments": 1,
        "scenario": rec.get("scenario", None),
        "instance": rec.get("instance", None),
    })



print(f"[fuzzy] train windows leidos: {len(all_train_segments)} | test windows: {len(all_test_segments)}")
print(f"[fuzzy] columnas de sensores: {sensor_cols}")
print(f"[fuzzy] largo temporal detectado: {WINDOW_SIZE_ROWS}")


# ============================================================
#  Preparacion de los datos
# ============================================================

# ---------- 1) Convertir listas a numpy  ----------
if len(all_train_segments) == 0 or len(all_test_segments) == 0:
    raise RuntimeError("No hay ventanas de train o de test tras filtrar TRANSITION/unknown.")

X_train = np.stack(all_train_segments, axis=0)  # (N_train, T, F)
X_test  = np.stack(all_test_segments,  axis=0)  # (N_test,  T, F)

labels_train = [d["label"] for d in all_train_labels]
labels_test  = [d["label"] for d in all_test_labels]

N_train, T_train, F = X_train.shape
N_test,  T_test,  F_test = X_test.shape

print(f"[model] X_train shape: {X_train.shape}  | X_test shape: {X_test.shape}")

if T_train != WINDOW_SIZE_ROWS:
    print(f"[WARN] T_train={T_train} != WINDOW_SIZE_ROWS={WINDOW_SIZE_ROWS}")
if T_test != WINDOW_SIZE_ROWS:
    print(f"[WARN] T_test={T_test} != WINDOW_SIZE_ROWS={WINDOW_SIZE_ROWS}")
if F_test != F:
    raise RuntimeError(f"Train y test tienen distinto no de canales: F_train={F}, F_test={F_test}")

# ---------- 2) NaN -> 0 ----------
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

# ---------- 3) Normalizacion condicional (Z-SCORE) ----------

if DO_NORMALIZATION:
    # min_ch = X_train.min(axis=(0, 1))  # (F,)
    # max_ch = X_train.max(axis=(0, 1))  # (F,)

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    for j in range(F):
        # mn = min_ch[j]
        # mx = max_ch[j]


        # if mn >= 0.0 and mx <= 1.0:
        #     # mantener tal cual
        #     continue

        # print("Normalizing ", j)
        

        channel_train = X_train[:, :, j]  # (N,T)
        mean_j = channel_train.mean()
        std_j  = channel_train.std()

        if std_j < 1e-6:
            std_j = 1.0  # evitar explosiones si es constante

       
        X_train_norm[:, :, j] = (X_train[:, :, j] - mean_j) / std_j
        X_test_norm[:, :, j]  = (X_test[:, :, j]  - mean_j) / std_j

    # Sustituimos los originales
    X_train = X_train_norm
    X_test  = X_test_norm

    print("[model] Normalizacion Z-SCORE condicional aplicada canal por canal.")

    # La logica original de la notebook de "mantener tal cual" no parece aplicarse
    # en la implementacion de Z-SCORE, asi que se ha simplificado.
    # El mensaje de "Normalizacion por canal a [0,1] aplicada" se omite.
else:
    print("[model] Normalizacion Z-SCORE omitida (usar --normalize para activarla).")


# ---------- 4) Codificar labels a indices ----------
all_labels_str = sorted(set(labels_train + labels_test))
label2idx = {lab: i for i, lab in enumerate(all_labels_str)}
idx2label = {i: lab for lab, i in label2idx.items()}

y_train = np.array([label2idx[l] for l in labels_train], dtype=np.int64)
y_test  = np.array([label2idx[l] for l in labels_test],  dtype=np.int64)

print("[model] Clases:")
for i, lab in idx2label.items():
    print(f"  {i}: {lab}")

num_classes = len(all_labels_str)
seq_len = T_train
n_channels = F

# ============================================================
#  Modelos de Referencia: KNN y SVM
# ============================================================
X_train_flat = X_train.reshape(N_train, -1)
X_test_flat  = X_test.reshape(N_test,  -1)

print("\n[baselines] X_train_flat:", X_train_flat.shape, "| X_test_flat:", X_test_flat.shape)

# --- KNN ---
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    n_jobs=-1,
)

knn.fit(X_train_flat, y_train)

y_pred_knn_train = knn.predict(X_train_flat)
y_pred_knn_test  = knn.predict(X_test_flat)

knn_train_acc = accuracy_score(y_train, y_pred_knn_train)
knn_train_f1  = f1_score(y_train, y_pred_knn_train, average="macro")
knn_train_f2  = f1_score(y_train, y_pred_knn_train, average="weighted")

knn_test_acc = accuracy_score(y_test, y_pred_knn_test)
knn_test_f1  = f1_score(y_test, y_pred_knn_test, average="macro")
knn_test_f2  = f1_score(y_test, y_pred_knn_test, average="weighted")

print("[KNN] "
      f"train_acc={knn_train_acc:.4f} | train_f1={knn_train_f1:.4f} | train_f2={knn_train_f2:.4f} | "
      f"test_acc={knn_test_acc:.4f} | test_f1={knn_test_f1:.4f} | test_f2={knn_test_f2:.4f}")


# --- SVM ---
svm_clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
)

print("[SVM] Entrenando SVC (esto puede tardar algo si N_train es grande)...")
svm_clf.fit(X_train_flat, y_train)

y_pred_svm_train = svm_clf.predict(X_train_flat)
y_pred_svm_test  = svm_clf.predict(X_test_flat)

svm_train_acc = accuracy_score(y_train, y_pred_svm_train)
svm_train_f1  = f1_score(y_train, y_pred_svm_train, average="macro")
svm_train_f2  = f1_score(y_train, y_pred_svm_train, average="weighted")

svm_test_acc = accuracy_score(y_test, y_pred_svm_test)
svm_test_f1  = f1_score(y_test, y_pred_svm_test, average="macro")
svm_test_f2  = f1_score(y_test, y_pred_svm_test, average="weighted")

print("[SVM] "
      f"train_acc={svm_train_acc:.4f} | train_f1={svm_train_f1:.4f} | train_f2={svm_train_f2:.4f} | "
      f"test_acc={svm_test_acc:.4f} | test_f1={svm_test_f1:.4f} | test_f2={svm_test_f2:.4f}")


# ============================================================
#  Modelo CNN+LSTM
# ============================================================

# ---------- 5) Pasar a tensores PyTorch ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[model] Usando device: {device}")

X_train_t = torch.from_numpy(X_train).float()
X_test_t  = torch.from_numpy(X_test).float()
y_train_t = torch.from_numpy(y_train)
y_test_t  = torch.from_numpy(y_test)

batch_size = 16

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)


# ---------- 6) Definir modelo CNN+LSTM "similar" ----------
class CNNLSTM(nn.Module):
    """
    CNN + doble LSTM estilo MARBLE:
      - 3 Conv1d sobre el eje temporal
      - LSTM con 2 capas (num_layers=2), 512 unidades
      - FC de 64 neuronas + capa final a num_classes
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # Conv sobre (B, F, T)
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128,        out_channels=256, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Doble LSTM (2 capas)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,      # dropout entre capas LSTM
            bidirectional=False
        )

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)            # -> (B, F, T) para Conv1d

        x = self.relu(self.conv1(x))     # (B, 128, T)
        x = self.relu(self.conv2(x))     # (B, 256, T)

        x = x.transpose(1, 2)            # -> (B, T, 512) para LSTM

        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers=2, B, 512) -> nos quedamos con la ultima capa
        h_last = h_n[-1]                 # (B, 512)

        h_last = self.dropout(h_last)
        x = self.relu(self.fc1(h_last))  # (B, 64)
        logits = self.fc2(x)             # (B, n_classes)
        return logits

class OnlyLSTM(nn.Module):
    """
    Solo LSTM estilo "parte recurrente" del CNN+LSTM:
      - LSTM con 2 capas (num_layers=2), 512 unidades
      - FC de 128 neuronas + capa final a num_classes
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        self.lstm = nn.LSTM(
            input_size=n_channels,   # ahora la entrada es directamente F (canales)
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,             # dropout entre capas LSTM
            bidirectional=False
        )

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]              # (B, 512) última capa

        h_last = self.dropout(h_last)
        x = self.relu(self.fc1(h_last))
        logits = self.fc2(x)
        return logits


#model = OnlyLSTM(n_channels=n_channels, n_classes=num_classes).to(device)

model = CNNLSTM(n_channels=n_channels, n_classes=num_classes).to(device)
counts = Counter(y_train.tolist())
total = len(y_train)

# peso_i = total / (num_classes * count_i)
class_weights = []
for i in range(num_classes):
    cnt = counts[i]
    if cnt == 0:
        w = 0.0
    else:
        w = total / (num_classes * cnt)
    class_weights.append(w)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("\n[class balance] Pesos por clase:")
for i, w in enumerate(class_weights.cpu().numpy()):
    print(f"  Clase {i} ({idx2label[i]}): peso={w:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

print(model)


# ---------- 7) Entrenamiento + evaluacion simple ----------
num_epochs = 40

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)

    # --- Evaluacion en train ---
    model.eval()
    train_preds = []
    train_true  = []

    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            train_preds.append(preds.cpu().numpy())
            train_true.append(yb.cpu().numpy())

    train_preds = np.concatenate(train_preds)
    train_true  = np.concatenate(train_true)

    train_acc = accuracy_score(train_true, train_preds)
    train_f1  = f1_score(train_true, train_preds, average="macro")
    train_f2  = f1_score(train_true, train_preds, average="weighted")
    
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # ?? EVALUACION TEST (igual que antes)
    # ---------------------------------------------------------
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)

    test_acc = accuracy_score(all_true, all_preds)
    test_f1  = f1_score(all_true, all_preds, average="macro")
    test_f2  = f1_score(all_true, all_preds, average="weighted")

    print(f"[Epoch {epoch:02d}] "
          f"loss={avg_train_loss:.4f} | "
          f"train_acc={train_acc:.4f} | train_f1={train_f1:.4f} | train_f2={train_f2:.4f} | "
          f"test_acc={test_acc:.4f} | test_f1={test_f1:.4f} | test_f2={test_f2:.4f}")