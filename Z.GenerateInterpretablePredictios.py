from pathlib import Path
import json
import numpy as np
import joblib
from xgboost import XGBClassifier


import re

def format_stats_feature(feat_name: str, value: float) -> str:
    """
    Stats naming soportado:
      "#p2#3#4@mean@magnetometer_x"
      "#p1@void@HALL"
      "#p3@number changes@gyr_rw_y"
    """
    try:
        if not feat_name.startswith("#"):
            raise ValueError("not a stats feature")

        body = feat_name[1:]
        group_label, stat_name, sensor = body.split("@", 2)  # por si stat_name tiene espacios

        # group_label: "p2#3#4" o "p3"
        if not group_label.startswith("p"):
            span = "[?]"
        else:
            label_wo_p = group_label[1:]              # "2#3#4"
            parts = [p for p in label_wo_p.split("#") if p.strip() != ""]
            idxs = [int(p) for p in parts] if parts else []

            if len(idxs) == 0:
                span = "[?]"
            elif len(idxs) == 1:
                span = f"[{idxs[0]}s]"
            else:
                span = f"[{min(idxs)}s,{max(idxs)}s]"

        stat_clean = stat_name.strip().lower()
        sensor_clean = sensor.strip()

        # Puedes ajustar wording aquí
        return f"{stat_clean} of {sensor_clean} between {span} is {value:.2f}"

    except Exception:
        return f"{feat_name} is {value:.2f}"


def format_logic_feature(feat_name: str, value: float) -> str:
    """
    Logic naming soportado:
      "[None] loc_toileting [is Low] [toward the end]"
      "[Many] gyr_la_y [is Low] [toward the end]"
    """
    try:
        cleaned = feat_name.replace("[", "").replace("]", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return f"{cleaned} ({value:.2f})"
    except Exception:
        return f"{feat_name} ({value:.2f})"


def format_feature_auto(feat_name: str, value: float) -> str:
    """
    Decide si es stats o logic según patrón del nombre.
    """
    # stats: empieza por "#p"
    if isinstance(feat_name, str) and feat_name.startswith("#p"):
        return format_stats_feature(feat_name, value)

    # logic: contiene corchetes tipo "[...]" (heurística)
    if isinstance(feat_name, str) and "[" in feat_name and "]" in feat_name:
        return format_logic_feature(feat_name, value)

    # fallback
    return f"{feat_name} ({value:.2f})"
    
# ----------------------------
# 1) Load npz (by_type_v1)
# ----------------------------
def load_features_npz_by_type(npz_path: str):
    data = np.load(str(npz_path), allow_pickle=True)

    X_wear = data["X_wear_f16"]
    X_bin  = data["X_bin_f16"]
    y      = data["y"].astype(np.int32)

    meta_json = data["meta_json"].item()
    meta = json.loads(meta_json)

    # None safety
    if isinstance(X_wear, np.ndarray) and X_wear.dtype == object and X_wear.shape == ():
        X_wear = None
    if isinstance(X_bin, np.ndarray) and X_bin.dtype == object and X_bin.shape == ():
        X_bin = None

    # back to float32 for sklearn
    if X_wear is not None:
        X_wear = X_wear.astype(np.float32)
    if X_bin is not None:
        X_bin = X_bin.astype(np.float32)

    wear_cols = meta.get("wear_cols", [])
    bin_cols  = meta.get("bin_cols", [])

    return X_wear, X_bin, y, wear_cols, bin_cols, meta

def flatten_by_type(X_wear_1, X_bin_1):
    """(Cw,Dw) + (Cb,Db) -> (1, D)"""
    parts = []
    if X_wear_1 is not None:
        parts.append(X_wear_1.reshape(1, -1))
    if X_bin_1 is not None:
        parts.append(X_bin_1.reshape(1, -1))
    if not parts:
        raise ValueError("No hay features (X_wear y X_bin son None).")
    return np.concatenate(parts, axis=1)

# ----------------------------
# 2) Load model bundle
# ----------------------------
def load_bundle(bundle_dir: str):
    p = Path(bundle_dir)

    knn  = joblib.load(p / "knn.joblib")
    svm  = joblib.load(p / "svm_linear.joblib")
    tree = joblib.load(p / "tree_extratrees.joblib")

    xgb = XGBClassifier()
    xgb.load_model(str(p / "xgb.json"))

    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))

    # mappings
    id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
    label2id = {k: int(v) for k, v in meta.get("label2id", {}).items()} if "label2id" in meta else None

    return knn, svm, tree, xgb, meta, id2label, label2id

def svm_margin_to_proba(margin_row, temperature=2.0):
    z = np.asarray(margin_row, dtype=float) / float(temperature)
    z = z - np.max(z)
    expz = np.exp(z)
    return expz / expz.sum()

# ----------------------------
# 3) Predict one test sample
# ----------------------------
def predict_index_test(
    test_npz: str,
    bundle_dir: str,
    index_test: int,
    svm_temperature: float = 2.0,
):
    Xw, Xb, y, wear_cols, bin_cols, meta_npz = load_features_npz_by_type(test_npz)
    knn, svm, tree, xgb, meta_bundle, id2label, _ = load_bundle(bundle_dir)

    n = y.shape[0]
    if index_test < 0 or index_test >= n:
        raise IndexError(f"INDEX_TEST={index_test} fuera de rango [0, {n-1}]")

    # sample tensors
    Xw_i = None if Xw is None else Xw[index_test]   # (Cw, Dw)
    Xb_i = None if Xb is None else Xb[index_test]   # (Cb, Db)
    X_i  = flatten_by_type(Xw_i, Xb_i)              # (1, D)

    y_true = int(y[index_test])

    # --- KNN
    y_knn = int(knn.predict(X_i)[0])
    knn_proba = knn.predict_proba(X_i)[0] if hasattr(knn, "predict_proba") else None

    # --- SVM LinearSVC (no proba nativa)
    y_svm = int(svm.predict(X_i)[0])
    margin = svm.decision_function(X_i)[0]
    svm_proba = svm_margin_to_proba(margin, temperature=svm_temperature)

    # --- Tree (ExtraTrees)
    y_tree = int(tree.predict(X_i)[0])
    tree_proba = tree.predict_proba(X_i)[0] if hasattr(tree, "predict_proba") else None

    # --- XGB
    y_xgb = int(xgb.predict(X_i)[0])
    xgb_proba = xgb.predict_proba(X_i)[0]

    def lbl(cid: int) -> str:
        return id2label.get(cid, str(cid))

    out = {
        "INDEX_TEST": index_test,
        "y_true_id": y_true,
        "y_true": lbl(y_true),
        "pred": {
            "knn": {"id": y_knn, "label": lbl(y_knn), "proba": None if knn_proba is None else knn_proba.tolist()},
            "svm": {"id": y_svm, "label": lbl(y_svm), "proba": svm_proba.tolist()},
            "tree": {"id": y_tree, "label": lbl(y_tree), "proba": None if tree_proba is None else tree_proba.tolist()},
            "xgb": {"id": y_xgb, "label": lbl(y_xgb), "proba": xgb_proba.tolist()},
        },
        "shapes": {
            "X_wear": None if Xw_i is None else list(Xw_i.shape),
            "X_bin":  None if Xb_i is None else list(Xb_i.shape),
            "X_flat": list(X_i.shape),
        },
        "meta_npz": {
            "wear_cols": wear_cols,
            "bin_cols": bin_cols,
            "logic_step": meta_npz.get("logic_step"),
            "stats_step": meta_npz.get("stats_step"),
            "T5": meta_npz.get("T5"),
            "SW": meta_npz.get("SW"),
        }
    }

    return out
    
    
def load_features_npz_flat(npz_path: str):
    data = np.load(str(npz_path), allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    feature_names = list(data["feature_names"].tolist()) if "feature_names" in data.files else None

    meta = {}
    if "meta_json" in data.files:
        meta = json.loads(data["meta_json"].item())

    return X, y, feature_names, meta

    
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def eval_random_subset_flat(test_flat, bundle_dir, N=100, seed=42):
    X_test, y_test, _, _ = load_features_npz_flat(test_flat)
    knn, svm, tree, xgb, _, _, _ = load_bundle(bundle_dir)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(y_test), size=min(N, len(y_test)), replace=False)

    y_true = y_test[idxs]
    y_knn  = knn.predict(X_test[idxs])
    y_svm  = svm.predict(X_test[idxs])
    y_tree = tree.predict(X_test[idxs])
    y_xgb  = xgb.predict(X_test[idxs])

    def rep(name, yp):
        print(f"[{name:5s}] acc={accuracy_score(y_true, yp):.3f} | f1_macro={f1_score(y_true, yp, average='macro'):.3f}")

    print(f"\n=== Eval flat sobre {len(idxs)} muestras ===")
    rep("KNN", y_knn)
    rep("SVM", y_svm)
    rep("Tree", y_tree)
    rep("XGB", y_xgb)

def knn_get_neighbors(knn, X_test, index):
    # indices de los vecinos
    dists, idxs = knn.kneighbors(X_test[index].reshape(1, -1), return_distance=True)
    return idxs[0], dists[0]
    


def aggregate_by_sensor(contrib_mean, feature_names):
    agg = defaultdict(list)
    for v, name in zip(contrib_mean, feature_names):
        sensor = name.split("@")[0]
        agg[sensor].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}
    
from collections import defaultdict
import numpy as np

import numpy as np

def knn_feature_contributions(
    knn, 
    X_train, y_train,
    X_query, y_query,
    feature_names,
    index_query,
    top_k_features=20
):
    x = X_query[index_query]

    dists, idxs = knn.kneighbors(x.reshape(1, -1), return_distance=True)
    idxs = idxs[0]
    dists = dists[0]

    neighbors = X_train[idxs]
    contrib = (neighbors - x)**2
    contrib_mean = contrib.mean(axis=0)

    order = np.argsort(contrib_mean)        # similares
    order_large = np.argsort(-contrib_mean) # diferentes

    return {
        "index_query": int(index_query),
        "y_query": int(y_query[index_query]),
        "neighbors_idx_train": idxs.tolist(),
        "neighbors_y_train": [int(y_train[i]) for i in idxs],
        "neighbors_dist": dists.tolist(),

        "top_similar_features": [
            {"feature": feature_names[i], "mean_sq_diff": float(contrib_mean[i])}
            for i in order[:top_k_features]
        ],

        "top_different_features": [
            {"feature": feature_names[i], "mean_sq_diff": float(contrib_mean[i])}
            for i in order_large[:top_k_features]
        ],

        "contrib_mean": contrib_mean,
    }

def knn_top_sensors(contrib_mean, feature_names, top_k=5):
    agg = defaultdict(list)
    for v, name in zip(contrib_mean, feature_names):
        sensor = name.split("@")[0] if "@" in name else name
        agg[sensor].append(v)
    sensor_score = {k: float(np.mean(v)) for k, v in agg.items()}
    # menor = más parecido (más “relevante” en similitud)
    return sorted(sensor_score.items(), key=lambda kv: kv[1])[:top_k]

import numpy as np

def svm_explain_one(
    svm,
    x_1d,                 # (D,)
    feature_names,
    id2label=None,
    top_k=15,
):
    """
    Explica una predicción de LinearSVC.
    Devuelve:
      - clase predicha
      - top contribuciones para la clase predicha (w*x)
      - top contribuciones por contraste pred vs segunda ( (w_pred-w_2nd)*x )
    """
    x = x_1d.reshape(1, -1)

    scores = svm.decision_function(x)  # (C,) o (1,C)
    scores = np.asarray(scores).reshape(-1)
    pred = int(np.argmax(scores))

    # ranking de clases por score
    order = np.argsort(-scores)
    second = int(order[1]) if len(order) > 1 else pred

    # pesos
    W = svm.coef_              # (C, D) en OVR
    b = svm.intercept_         # (C,)

    # contribuciones por feature para clase pred: w_pred * x
    w_pred = W[pred]
    contrib_pred = w_pred * x_1d  # (D,)

    # contribuciones contraste pred vs second: (w_pred-w_2nd) * x
    w_diff = W[pred] - W[second]
    contrib_diff = w_diff * x_1d

    def top_signed(contrib, k):
        # top positivos (a favor)
        pos_idx = np.argsort(-contrib)[:k]
        # top negativos (en contra)
        neg_idx = np.argsort(contrib)[:k]
        return (
            [{"feature": feature_names[i], "value": float(contrib[i])} for i in pos_idx],
            [{"feature": feature_names[i], "value": float(contrib[i])} for i in neg_idx],
        )

    pos_pred, neg_pred = top_signed(contrib_pred, top_k)
    pos_diff, neg_diff = top_signed(contrib_diff, top_k)

    def lbl(c):
        return id2label.get(int(c), str(int(c))) if isinstance(id2label, dict) else str(int(c))

    return {
        "scores": {lbl(i): float(scores[i]) for i in range(len(scores))},
        "pred_id": pred,
        "pred_label": lbl(pred),
        "second_id": second,
        "second_label": lbl(second),
        "top_pred_positive": pos_pred,
        "top_pred_negative": neg_pred,
        "top_contrast_positive": pos_diff,
        "top_contrast_negative": neg_diff,
    }


import numpy as np

def svm_explain_one_zscore(
    svm,
    x_1d,                 # (D,)
    mu, sigma,            # (D,) de TRAIN
    feature_names,
    id2label=None,
    top_k=15
):
    x = x_1d.reshape(1, -1)
    scores = np.asarray(svm.decision_function(x)).reshape(-1)
    pred = int(np.argmax(scores))
    order = np.argsort(-scores)
    second = int(order[1]) if len(order) > 1 else pred

    W = svm.coef_          # (C, D)
    b = svm.intercept_     # (C,)

    # z-score SOLO para ranking
    x_norm = (x_1d - mu) / sigma

    # contribuciones para clase predicha
    contrib_pred = W[pred] * x_norm

    # contraste pred vs second
    contrib_diff = (W[pred] - W[second]) * x_norm

    def top_signed(arr, k):
        arr = np.asarray(arr)
        pos_idx = np.argsort(-arr)[:k]
        neg_idx = np.argsort(arr)[:k]
        return (
            [{"feature": feature_names[i], "value": float(arr[i])} for i in pos_idx],
            [{"feature": feature_names[i], "value": float(arr[i])} for i in neg_idx],
        )

    pos_pred, neg_pred = top_signed(contrib_pred, top_k)
    pos_diff, neg_diff = top_signed(contrib_diff, top_k)

    def lbl(c):
        return id2label.get(int(c), str(int(c))) if isinstance(id2label, dict) else str(int(c))

    return {
        "pred_id": pred,
        "pred_label": lbl(pred),
        "second_id": second,
        "second_label": lbl(second),
        "scores": {lbl(i): float(scores[i]) for i in range(len(scores))},
        "top_pred_positive": pos_pred,
        "top_pred_negative": neg_pred,
        "top_contrast_positive": pos_diff,
        "top_contrast_negative": neg_diff,
    }


def svm_explain_one(
    svm,
    x_1d,                 # (D,)
    feature_names,
    id2label=None,
    top_k=15,
):
    """
    Explica una predicción de LinearSVC.
    Devuelve:
      - clase predicha
      - top contribuciones para la clase predicha (w*x)
      - top contribuciones por contraste pred vs segunda ( (w_pred-w_2nd)*x )
    """
    x = x_1d.reshape(1, -1)

    scores = svm.decision_function(x)  # (C,) o (1,C)
    scores = np.asarray(scores).reshape(-1)
    pred = int(np.argmax(scores))

    # ranking de clases por score
    order = np.argsort(-scores)
    second = int(order[1]) if len(order) > 1 else pred

    # pesos
    W = svm.coef_              # (C, D) en OVR
    b = svm.intercept_         # (C,)

    # contribuciones por feature para clase pred: w_pred * x
    w_pred = W[pred]
    contrib_pred = w_pred * x_1d  # (D,)

    # contribuciones contraste pred vs second: (w_pred-w_2nd) * x
    w_diff = W[pred] - W[second]
    contrib_diff = w_diff * x_1d

    def top_signed(contrib, k):
        # top positivos (a favor)
        pos_idx = np.argsort(-contrib)[:k]
        # top negativos (en contra)
        neg_idx = np.argsort(contrib)[:k]
        return (
            [{"feature": feature_names[i], "value": float(contrib[i])} for i in pos_idx],
            [{"feature": feature_names[i], "value": float(contrib[i])} for i in neg_idx],
        )

    pos_pred, neg_pred = top_signed(contrib_pred, top_k)
    pos_diff, neg_diff = top_signed(contrib_diff, top_k)

    def lbl(c):
        return id2label.get(int(c), str(int(c))) if isinstance(id2label, dict) else str(int(c))

    return {
        "scores": {lbl(i): float(scores[i]) for i in range(len(scores))},
        "pred_id": pred,
        "pred_label": lbl(pred),
        "second_id": second,
        "second_label": lbl(second),
        "top_pred_positive": pos_pred,
        "top_pred_negative": neg_pred,
        "top_contrast_positive": pos_diff,
        "top_contrast_negative": neg_diff,
    }
    
import numpy as np

def tree_explain_one(
    tree_model,
    x_1d,                   # (D,)
    feature_names,
    id2label=None
):
    """
    Explicación local estilo C4.5:
    - recorrido root → leaf
    - reglas if–then
    """
    x = x_1d.reshape(1, -1)

    # si es ExtraTrees, cogemos el primer árbol
    if hasattr(tree_model, "estimators_"):
        tree = tree_model.estimators_[0].tree_
        classes = tree_model.classes_
    else:
        tree = tree_model.tree_
        classes = tree_model.classes_

    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value

    node = 0
    path = []

    while children_left[node] != children_right[node]:
        feat_idx = feature[node]
        feat_name = feature_names[feat_idx]
        thr = threshold[node]
        val = x_1d[feat_idx]

        if val <= thr:
            decision = "<="
            next_node = children_left[node]
        else:
            decision = ">"
            next_node = children_right[node]

        path.append({
            "feature": feat_name,
            "value": float(val),
            "threshold": float(thr),
            "decision": decision
        })

        node = next_node

    # hoja
    leaf_value = value[node][0]
    pred_id = int(np.argmax(leaf_value))
    pred_label = id2label.get(int(classes[pred_id]), str(classes[pred_id])) \
        if isinstance(id2label, dict) else str(classes[pred_id])

    return {
        "pred_id": int(classes[pred_id]),
        "pred_label": pred_label,
        "path": path,
        "leaf_distribution": leaf_value.tolist()
    }
def print_tree_explanation(exp):
    print(f"Prediction: {exp['pred_label']}")
    print("Decision path:")
    for step in exp["path"]:
        print(
            " IF",
            format_feature_auto(step["feature"], step["value"]),
            step["decision"],
            f"{step['threshold']:.3f}"
        )
    print(" THEN → class distribution at leaf:", exp["leaf_distribution"])
    
import xgboost as xgb_lib

def xgb_explain_one(
    xgb_model,          # XGBClassifier ya entrenado/cargado
    x_1d,               # (D,)
    feature_names,      # list[str] len D
    id2label=None,
    top_k=15
):
    booster = xgb_model.get_booster()

    # asegurar nombres de features en el booster (opcional pero recomendable)
    safe_names = [f"f{i}" for i in range(len(feature_names))]

    # Asegura nombres seguros en booster
    booster.feature_names = safe_names

    dmat = xgb_lib.DMatrix(
        x_1d.reshape(1, -1),
        feature_names=safe_names
    )

    # predicción normal
    proba = xgb_model.predict_proba(x_1d.reshape(1, -1))[0]  # (C,)
    pred = int(np.argmax(proba))
    order = np.argsort(-proba)
    second = int(order[1]) if len(order) > 1 else pred

    # contribuciones tipo SHAP del booster
    # multiclass suele devolver (1, C, D+1) o (C, D+1) según versión
    contrib = booster.predict(dmat, pred_contribs=True)

    contrib = np.asarray(contrib)

    # Normalizamos shapes:
    # - si es (1, C, D+1) -> quitamos batch
    if contrib.ndim == 3 and contrib.shape[0] == 1:
        contrib = contrib[0]  # (C, D+1)
    # - si es (C, D+1) ok
    # - si es (1, D+1) (binario) -> lo tratamos como C=1
    if contrib.ndim == 2 and contrib.shape[0] == 1:
        contrib = contrib  # (1, D+1)

    # para la clase predicha:
    contrib_pred = contrib[pred]          # (D+1,)
    bias_pred = float(contrib_pred[-1])
    feat_contrib_pred = contrib_pred[:-1] # (D,)

    # contraste pred vs second:
    contrib_second = contrib[second]
    feat_contrib_diff = feat_contrib_pred - contrib_second[:-1]

    def top_signed(arr, k, eps=1e-2):
        arr = np.asarray(arr)
        valid = np.where(np.abs(arr) >= eps)[0]
        if len(valid) == 0:
            return ([], [])

        # ordenar solo índices válidos
        pos_idx = valid[np.argsort(-arr[valid])][:k]
        neg_idx = valid[np.argsort(arr[valid])][:k]

        return (
            [{"feature": feature_names[i], "value": float(arr[i])} for i in pos_idx],
            [{"feature": feature_names[i], "value": float(arr[i])} for i in neg_idx],
        )

    pos_pred, neg_pred = top_signed(feat_contrib_pred, top_k)
    pos_diff, neg_diff = top_signed(feat_contrib_diff, top_k)

    def lbl(c):
        return id2label.get(int(c), str(int(c))) if isinstance(id2label, dict) else str(int(c))

    return {
        "pred_id": pred,
        "pred_label": lbl(pred),
        "second_id": second,
        "second_label": lbl(second),
        "proba": {lbl(i): float(proba[i]) for i in range(len(proba))},
        "bias_pred": bias_pred,
        "top_pred_positive": pos_pred,
        "top_pred_negative": neg_pred,
        "top_contrast_positive": pos_diff,
        "top_contrast_negative": neg_diff,
    }
    
def print_xgb_explanation(xgb_exp, top_n=10):
    print("Pred:", xgb_exp["pred_label"], "| Second:", xgb_exp["second_label"])
    print("Bias (base):", xgb_exp["bias_pred"])

    print("\nTop features pushing TOWARD predicted class (SHAP contribs):")
    for it in xgb_exp["top_pred_positive"][:top_n]:
        print(" +", format_feature_auto(it["feature"], it["value"]))

    print("\nTop features pushing AGAINST predicted class (SHAP contribs):")
    for it in xgb_exp["top_pred_negative"][:top_n]:
        print(" -", format_feature_auto(it["feature"], it["value"]))

    #print("\nTop features pushing Pred OVER Second (contrib_pred - contrib_second):")
    #for it in xgb_exp["top_contrast_positive"][:top_n]:
    #    print(" >", format_feature_auto(it["feature"], it["value"]))

    #print("\nTop features pushing Second OVER Pred:")
    #for it in xgb_exp["top_contrast_negative"][:top_n]:
    #    print(" <", format_feature_auto(it["feature"], it["value"]))

    
if __name__ == "__main__":
    OUT_DIR = "data/toon_mhealth"
    test_flat = str(Path(OUT_DIR) / "features_test_flat.npz")
    bundle_dir = str(Path(OUT_DIR) / "models_bundle")

    import sys
    INDEX_TEST = int(sys.argv[1])

    
    train_flat = str(Path(OUT_DIR) / "features_train_flat.npz")
    test_flat  = str(Path(OUT_DIR) / "features_test_flat.npz")

    X_train, y_train, feature_names_tr, _ = load_features_npz_flat(train_flat)
    X_test,  y_test,  feature_names_te, _ = load_features_npz_flat(test_flat)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-9

    # usa los feature_names del train (deberían ser iguales)
    feature_names = feature_names_tr




    # 2) cargar modelos
    knn, svm, tree, xgb, meta_bundle, id2label, _ = load_bundle(bundle_dir)

    # 3) predecir una muestra
    x = X_test[INDEX_TEST].reshape(1, -1)
    y_true = int(y_test[INDEX_TEST])

    pred_knn  = int(knn.predict(x)[0])
    pred_svm  = int(svm.predict(x)[0])
    pred_tree = int(tree.predict(x)[0])
    pred_xgb  = int(xgb.predict(x)[0])

    def lbl(i): return id2label.get(int(i), str(int(i)))

    print(json.dumps({
        "INDEX_TEST": INDEX_TEST,
        "y_true_id": y_true,
        "y_true": lbl(y_true),
        "pred": {
            "knn":  {"id": pred_knn,  "label": lbl(pred_knn)},
            "svm":  {"id": pred_svm,  "label": lbl(pred_svm)},
            "tree": {"id": pred_tree, "label": lbl(pred_tree)},
            "xgb":  {"id": pred_xgb,  "label": lbl(pred_xgb)},
        },
        "shapes": {
            "X_test": list(X_test.shape),
            "x": list(x.shape)
        }
    }, ensure_ascii=False, indent=2))
    
    eval_random_subset_flat(test_flat, bundle_dir, N=100, seed=42)

    #KNN
    print("KNN explanibility")
    exp = knn_feature_contributions(
        knn=knn,
        X_train=X_train, y_train=y_train,
        X_query=X_test,  y_query=y_test,
        feature_names=feature_names,
        index_query=INDEX_TEST,
        top_k_features=25
    )

    print(exp["top_similar_features"][:10])
    
    print([item["mean_sq_diff"] for item in exp["top_similar_features"][:5]])
    print([item["mean_sq_diff"] for item in exp["top_different_features"][:5]])
    
    print("Explain:")
    EPS_KNN = 1e-4


    print("\nKNN – MOST SIMILAR (informative)")
    for item in exp["top_similar_features"]:
            print(" *", format_feature_auto(item["feature"], item["mean_sq_diff"]))
    print("\nKNN – MOST DIFFERENT (informative)")
    for item in exp["top_different_features"]:
        if item["mean_sq_diff"] >= EPS_KNN:
            print(" *", format_feature_auto(item["feature"], item["mean_sq_diff"]))

    #top_sensors = knn_top_sensors(exp["contrib_mean"], feature_names, top_k=20)
    #print("Top sensores:", top_sensors)
    #for sensor_name, score in top_sensors:
    #    print(sensor_name)
    
    #SVM
    # SVM
    print("SVM explanibility")

    x_1d = X_test[INDEX_TEST]

    svm_exp = svm_explain_one_zscore(
        svm=svm,
        x_1d=x_1d,
        mu=mu, sigma=sigma,
        feature_names=feature_names,
        id2label=id2label,
        top_k=15
    )

    print("Pred:", svm_exp["pred_label"], "| Second:", svm_exp["second_label"])

    print("\nTop (zscore) pushing TOWARD predicted:")
    for it in svm_exp["top_pred_positive"]:
        print(" +", format_feature_auto(it["feature"], it["value"]))

    print("\nTop (zscore) pushing AGAINST predicted:")
    for it in svm_exp["top_pred_negative"]:
        print(" -", format_feature_auto(it["feature"], it["value"]))

    #print("\nTop features pushing Pred OVER Second ((w_pred-w_2nd)*x):")
    #for it in svm_exp["top_contrast_positive"]:
    #    print(" >", format_feature_auto(it["feature"], it["value"]))

    #print("\nTop features pushing Second OVER Pred ((w_pred-w_2nd)*x):")
    #for it in svm_exp["top_contrast_negative"]:
    #    print(" <", format_feature_auto(it["feature"], it["value"]))

    print("C4.5 explainability")

    x_1d = X_test[INDEX_TEST]

    tree_exp = tree_explain_one(
        tree_model=tree,          # o clf / extra_trees
        x_1d=x_1d,
        feature_names=feature_names,
        id2label=id2label
    )

    print_tree_explanation(tree_exp)
    
    print("XGBoost explainability")

    x_1d = X_test[INDEX_TEST]

    xgb_exp = xgb_explain_one(
        xgb_model=xgb,
        x_1d=x_1d,
        feature_names=feature_names,
        id2label=id2label,
        top_k=12
    )

    print_xgb_explanation(xgb_exp, top_n=10)
