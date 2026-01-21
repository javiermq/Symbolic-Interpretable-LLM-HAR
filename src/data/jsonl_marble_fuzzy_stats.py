# jsonl_marble_fuzzy_stats.py
"""
Dataset MARBLE que genera features fuzzy + estadísticas a partir de las ventanas crudas,
reutilizando el procesamiento de Z.UWB (fuzzy logic + stats).

Uso típico:

    from src.data.jsonl_marble_fuzzy_stats import JsonlMarbleFuzzyStats

    ds_tr = JsonlMarbleFuzzyStats(
        raw_path="data/marble/train_raw.jsonl",
        channels=[...],      # mismos canales que en tu cfg
        fs=fs,
        duration=duration,
    )

    sample = ds_tr[0]
    x = sample["X"]   # tensor (D,) con fuzzy+stats
    y = sample["y"]   # label int

Luego puedes hacer:
    MLP( x ) -> proyector -> LLaMA

No hay Chronos en este pipeline.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
import random


def _flatten_group(group, max_per_channel=1):
    if not isinstance(group, dict):
        return []

    phrases = []
    for _, sent_list in group.items():
        if not isinstance(sent_list, list):
            continue

        taken = 0
        for s in sent_list:
            if taken >= max_per_channel:
                break
            if not isinstance(s, str):
                continue
            s = s.strip()
            if not s:
                continue
            phrases.append(s)
            taken += 1
    return phrases


def build_lbm_description(
    desc_row: Dict[str, Any],
    max_per_channel: int = 1,
    section_headers: bool = True,
) -> str:
    """
    Construye un texto descriptivo a partir de una fila de descripciones.

    Soporta dos esquemas:

    1) Esquema clásico MARBLE: claves L / B / M
       - L: location
       - B: binary env/smartphone
       - M: wearable (IMU + magnetómetro)

    2) Esquema TOON / MHEALTH: claves W / B
       - W: wearables (IMU, etc.)
       - B: binarios (si los hubiera)

    Si detecta W/B sin L/M, usa solo W y B.
    Si detecta L/M, usa el esquema L/B/M original.
    """
    # ------------------------------
    # Detectar esquema de claves
    # ------------------------------
    has_W_or_B = ("W" in desc_row) or ("B" in desc_row)
    has_L_or_M = ("L" in desc_row) or ("M" in desc_row)

    sections: List[str] = []

    # =========================================================
    # CASO 1: solo W/B (TOON, MHEALTH, etc.)
    # =========================================================
    if has_W_or_B and not has_L_or_M:
        W = desc_row.get("W", {})
        B = desc_row.get("B", {})

        # ---- Wearable signals (W) ----
        w_phrases = _flatten_group(W, max_per_channel=max_per_channel)
        w_phrases = [p for p in w_phrases if isinstance(p, str)]
        random.shuffle(w_phrases)

        if w_phrases:
            if section_headers:
                sections.append("Wearable signals: " + "; ".join(w_phrases))
            else:
                sections.extend(w_phrases)

        # ---- Binary sensors (B) ----
        b_phrases = _flatten_group(B, max_per_channel=max_per_channel)
        b_phrases = [p for p in b_phrases if isinstance(p, str)]
        random.shuffle(b_phrases)

        if b_phrases:
            if section_headers:
                sections.append("Binary sensors: " + "; ".join(b_phrases))
            else:
                sections.extend(b_phrases)

        if section_headers:
            return "\n".join(sections)
        else:
            return " ".join(sections)

    # =========================================================
    # CASO 2: esquema clásico L/B/M (MARBLE original)
    # =========================================================
    L = desc_row.get("L", "")  # location
    B = desc_row.get("B", "")  # binary env/smartphone
    M = desc_row.get("M", "")  # wearable (IMU + magnetómetro)

    # -------- Location --------
    loc_phrases = _flatten_group(L, max_per_channel=max_per_channel)
    loc_phrases = [p for p in loc_phrases if isinstance(p, str)]
    random.shuffle(loc_phrases)

    if loc_phrases:
        if section_headers:
            sections.append("Location sensors: " + "; ".join(loc_phrases))
        else:
            sections.extend(loc_phrases)

    # -------- Environmental / Smartphone --------
    bin_phrases = _flatten_group(B, max_per_channel=max_per_channel)
    bin_phrases = [p for p in bin_phrases if isinstance(p, str)]
    random.shuffle(bin_phrases)

    if bin_phrases:
        if section_headers:
            sections.append(
                "Environmental and smartphone sensors: " + "; ".join(bin_phrases)
            )
        else:
            sections.extend(bin_phrases)

    # -------- Wearable IMU / Magnetometer --------
    mot_phrases = _flatten_group(M, max_per_channel=max_per_channel)
    mot_phrases = [p for p in mot_phrases if isinstance(p, str)]
    random.shuffle(mot_phrases)

    if mot_phrases:
        if section_headers:
            sections.append(
                "Wearable IMU and magnetometer: " + "; ".join(mot_phrases)
            )
        else:
            sections.extend(mot_phrases)

    if section_headers:
        return "\n".join(sections)
    else:
        return " ".join(sections)



import json
import numpy as np
import torch

def load_features_npz_by_type(path: str):
    z = np.load(path, allow_pickle=True)

    meta_raw = z["meta_json"].item()
    meta = json.loads(meta_raw if isinstance(meta_raw, str) else str(meta_raw))

    wear_cols = list(meta.get("wear_cols", []))
    bin_cols  = list(meta.get("bin_cols",  []))

    # --- WEAR ---
    X_wear = None
    if "X_wear_f16" in z.files and len(wear_cols) > 0:
        arr = z["X_wear_f16"]
        # descarta placeholders/objetos/vacíos
        if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype != object:
            X_wear = arr.astype(np.float32)
        else:
            X_wear = None

    # si no hay wear_cols, NO debe haber wear
    if len(wear_cols) == 0:
        X_wear = None

    # --- BIN ---
    X_bin = None
    if "X_bin_f16" in z.files and len(bin_cols) > 0:
        arr = z["X_bin_f16"]
        if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype != object:
            X_bin = arr.astype(np.float32)

    if (not bin_cols) or (X_bin is None):
        X_bin = None

    y = z["y"].astype(np.int64)

    # reshape BIN si viniera 2D
    if X_bin is not None and X_bin.ndim == 2:
        N, D = X_bin.shape
        Cb = len(bin_cols)
        if Cb <= 0:
            X_bin = None
        elif D % Cb == 0:
            X_bin = X_bin.reshape(N, Cb, D // Cb)
        elif Cb == 1:
            X_bin = X_bin.reshape(N, 1, D)
        else:
            raise ValueError(f"X_bin_f16 es 2D {X_bin.shape} pero bin_cols={Cb} no divide D={D}")

    # reshape WEAR si viniera 2D
    if X_wear is not None and X_wear.ndim == 2:
        N, D = X_wear.shape
        Cw = len(wear_cols)
        if Cw > 0 and D % Cw == 0:
            X_wear = X_wear.reshape(N, Cw, D // Cw)
        elif Cw == 1:
            X_wear = X_wear.reshape(N, 1, D)

    return X_wear, X_bin, y, meta




# ---------------------------------------------------------------------
# Utilidades básicas (basadas en jsonl_marble.py)
# ---------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    print("loading", path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pad_or_crop(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    x: np.ndarray (C, T_actual) → (C, target_len)
    """
    C, T = x.shape
    if T == target_len:
        return x
    if T > target_len:
        return x[:, :target_len]
    out = np.zeros((C, target_len), dtype=x.dtype)
    out[:, :T] = x
    return out




# ---------------------------------------------------------------------
# Detección de tipo de canal (reutiliza prefijos loc_/obj_ de MARBLE/UWB)
# ---------------------------------------------------------------------
from typing import List
# listas que se pueden rellenar dinámicamente desde el YAML
WEAR_SENSORS: List[str] = []
BINARY_SENSORS: List[str] = []

BINARY_TYPES = {"binary"}
# ---------------------------------------------------------------------
# Detección de tipo de canal (reutiliza prefijos loc_/obj_ de MARBLE/UWB)
# ---------------------------------------------------------------------




def infer_channel_type(ch_name: str) -> str:
    """
    Devuelve el tipo de canal usando, por este orden:

      1) Listas dinámicas WEAR_SENSORS / BINARY_SENSORS (rellenas desde YAML).
      2) Heurísticas por prefijo (loc_/obj_/env_/smartphone_*).

    Los tipos devueltos son strings como:
        - "binary"  → tratado como canal [0,1] (L2)
        - "wear"    → considerado continuo (L5)
        - "location"/"other" → continuo (L5)
    """
    # 1) Info dinámica desde sensor_cfg (si se ha rellenado)
    if ch_name in BINARY_SENSORS:
        return "binary"
    if ch_name in WEAR_SENSORS:
        return "wear"

    # 2) Heurística por nombres
    if ch_name.startswith("loc_"):
        return "location"
    if ch_name.startswith("obj_"):
        return "binary"
    if ch_name.startswith("env_") or ch_name.startswith("smartphone_"):
        return "binary"

    return "other"
def set_channel_type_map(channel_type_map: Dict[str, str]) -> None:
    """
    Rellena WEAR_SENSORS / BINARY_SENSORS a partir de un dict
    {nombre_canal -> tipo}, donde tipo ∈ {'WEAR','BINARY',...}.

    Ejemplo:
        {"acc_ch_x": "WEAR", "loc_cooking": "BINARY"}
    """
    global WEAR_SENSORS, BINARY_SENSORS

    WEAR_SENSORS = []
    BINARY_SENSORS = []

    for ch, tp in channel_type_map.items():
        tp_upper = str(tp).upper()

        if tp_upper == "WEAR":
            WEAR_SENSORS.append(ch)
        elif tp_upper == "BINARY":
            BINARY_SENSORS.append(ch)
        else:
            raise ValueError(
                f"[sensor_cfg] Tipo '{tp}' no permitido. "
                f"Solo se aceptan WEAR y BINARY. Canal: {ch}"
            )

    print("\n[jsonl_marble_fuzzy_stats] WEAR_SENSORS dinámicos:")
    for s in WEAR_SENSORS:
        print("   ", s)

    print("\n[jsonl_marble_fuzzy_stats] BINARY_SENSORS dinámicos:")
    for s in BINARY_SENSORS:
        print("   ", s)



# ---------------------------------------------------------------------
# Dataset principal: JsonlMarbleFuzzyStats
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Dataset principal: JsonlMarbleFuzzyStats
# ---------------------------------------------------------------------

class JsonlMarbleFuzzyStats(Dataset):
    def __init__(
        self,
        raw_path: str,
        channels: List[str],
        fs: float,
        duration: float,
        desc_path: Optional[str] = None,
        features_path: Optional[str] = None,   # 👈 NUEVO
        split="train",
        text_base: str = "",
        max_desc_per_channel: int = 1,
        activities: Optional[List[str]] = None,
        normalize_logic_to_01: bool = True,
        use_rag_description: bool = True, rag_prefix: str = "RAG: "
    ) -> None:
        """
        raw_path: JSONL de ventanas crudas tipo har_train_windows.jsonl
                  con campos "window", "channels" y "label"/"activity".
        desc_path: JSONL paralelo con campos L/B/M (descripciones toon).
        """
        super().__init__()
        
        

        self.split = split
        self.use_rag_description = bool(use_rag_description)
        self.rag_prefix = str(rag_prefix or "")
    
        self.raw_path = Path(raw_path)
        self.channels = list(channels)
        self.fs = float(fs)
        self.duration = float(duration)
        self.target_len = int(round(self.fs * self.duration))

        self.text_base = text_base
        self.max_desc_per_channel = max_desc_per_channel
        self.activities = list(activities) if activities is not None else None
        self.normalize_logic_to_01 = normalize_logic_to_01

        assert self.raw_path.exists(), f"raw_path no existe: {self.raw_path}"

        # 1) Cargar filas crudas (ventanas)
        # 1) Cargar filas crudas (para label/text/meta)
        self.rows = _load_jsonl(self.raw_path)

        labels_str: List[str] = []
        for idx, row in enumerate(self.rows):
            if "label" in row:
                lab_str = row["label"]
            elif "activity" in row:
                lab_str = row["activity"]
                row["label"] = row["activity"]
            else:
                raise KeyError(f"Fila idx={idx} no contiene ni 'label' ni 'activity'.")
            labels_str.append(lab_str)

        # 2) Labels (se mantienen por compatibilidad)
        unique_labels = sorted(set(labels_str))
        self.lab2id = {lab: i for i, lab in enumerate(unique_labels)}
        self.id2lab = {i: lab for lab, i in self.lab2id.items()}

        # 3) Features: LOAD-ONLY desde NPZ si features_path
        self._X = None
        self.feature_names = []

        if features_path is None:
            # (si quieres dejar fallback a generación, podrías mantenerlo;
            #  pero como has pedido quitar generación, aquí mejor error)
            raise ValueError("features_path es obligatorio en modo load-only.")
        else:
            Xw, Xb, y_npz, meta = load_features_npz_by_type(features_path)
            self.meta = meta

            self.wear_cols = list(meta.get("wear_cols", []))
            self.bin_cols  = list(meta.get("bin_cols", []))

            # sanity shapes
            if Xw is not None:
                assert Xw.ndim == 3 and Xw.shape[1] == len(self.wear_cols)
            if Xb is not None:
                assert Xb.ndim == 3 and Xb.shape[1] == len(self.bin_cols)

            # guarda arrays
            self._Xw = Xw
            self._Xb = Xb
            self._y  = y_npz

            # construye channel_types EN ORDEN GLOBAL self.channels
            wear_set = set(self.wear_cols)
            bin_set  = set(self.bin_cols)

            # checks fuertes
            cfg_set = set(self.channels)
            if cfg_set != (wear_set | bin_set):
                raise ValueError(
                    "channels(cfg) != wear_cols∪bin_cols(meta)\n"
                    f"cfg={len(cfg_set)} meta={len(wear_set|bin_set)}\n"
                    f"missing_in_meta={sorted(cfg_set - (wear_set|bin_set))}\n"
                    f"extra_in_meta={sorted((wear_set|bin_set) - cfg_set)}"
                )

            self.channel_types = ["WEAR" if ch in wear_set else "BINARY" for ch in self.channels]

            # mapeos global->pos en tensores por tipo
            self._wear_pos_global = [i for i,ch in enumerate(self.channels) if ch in wear_set]
            self._bin_pos_global  = [i for i,ch in enumerate(self.channels) if ch in bin_set]

            self._wear_index = {ch:i for i,ch in enumerate(self.wear_cols)}
            self._bin_index  = {ch:i for i,ch in enumerate(self.bin_cols)}

            # dims por tipo (para que el trainer las lea)
            self.d_w = None if Xw is None else int(Xw.shape[-1])
            self.d_b = None if Xb is None else int(Xb.shape[-1])

            # (opcional) “clavar” mapping de labels si viene
            id2label_npz = meta.get("id2label", None)
            if id2label_npz:
                id2label_npz = {int(k): str(v) for k, v in id2label_npz.items()}
                self.id2lab = dict(id2label_npz)
                self.lab2id = {v: k for k, v in self.id2lab.items()}



            print("[JsonlMarbleFuzzyStats] Xw:", None if self._Xw is None else self._Xw.shape,
              "| Xb:", None if self._Xb is None else self._Xb.shape,
              "| y:", self._y.shape)




        # 5) Descripciones L/B/M opcionales (toon)
        self.desc_rows = None
        if desc_path is not None:
            self.desc_path = Path(desc_path)
            assert self.desc_path.exists(), f"desc_path no existe: {self.desc_path}"
            self.desc_rows = _load_jsonl(self.desc_path)
            assert len(self.desc_rows) == len(self.rows), (
                f"rows fuzzy ({len(self.rows)}) y desc_rows ({len(self.desc_rows)}) "
                "no tienen la misma longitud."
            )

    # ----------------------------------------------
    # Construcción de fuzzy+stats globales
    # ----------------------------------------------
        # ----------------------------------------------
    # Construcción de fuzzy+stats globales
    # ----------------------------------------------


    def _build_text(self, desc_row: dict) -> str:
        """
        Construye el texto final a partir de las descripciones TOON.

        - Usa grupos 'W' (wearables) y 'B' (binarios).
        - Si un grupo está vacío (no hay frases no vacías), se elimina
          para que no aparezca el inicio de sección.
        """
        base = self.text_base or ""

        if desc_row is None or not isinstance(desc_row, dict):
            return base

        # ============================
        # 1) Lista de actividades
        # ============================
        activities_list = self.activities

        # Si no se han pasado actividades en el cfg, las obtenemos de id2lab
        if (not activities_list) and hasattr(self, "id2lab") and isinstance(self.id2lab, dict):
            activities_list = [self.id2lab[i] for i in sorted(self.id2lab.keys())]

        if activities_list:
            activities_str = "; ".join(
                f"{i+1}. <{name}>" for i, name in enumerate(activities_list)
            )
            #activities_block = f"Possible activities: {activities_str}"
            activities_block=""
            if base:
                base = base.rstrip() + "\n\n" + activities_block
            else:
                base = activities_block

        # ============================
        # 2) Normalizar claves a 'W' y 'B'
        #    (soportamos 'L' antiguo como 'W')
        # ============================
        src = dict(desc_row)  # copia defensiva

        W_group = src.get("W") or src.get("L") or {}
        B_group = src.get("B") or {}

        desc_clean = {"W": W_group, "B": B_group}

        # ============================
        # 3) Filtrar grupos vacíos
        # ============================
        def group_has_content(g: dict) -> bool:
            if not g:
                return False
            for _ch, texts in g.items():
                if not texts:
                    continue
                for t in texts:
                    if t is None:
                        continue
                    if str(t).strip() == "":
                        continue
                    return True
            return False

        import random

        final_desc = {}
        if group_has_content(desc_clean["W"]):
            final_desc["W"] = desc_clean["W"]
        if group_has_content(desc_clean["B"]):
            final_desc["B"] = desc_clean["B"]

        # Si no queda ningún grupo con contenido, devolvemos solo el base
        if not final_desc:
            return base

        # ============================================================
        # (NUEVO) Selección aleatoria de frases por canal
        # ============================================================
        BASE_SEED = 12345
        for grp in ["W", "B"]:
            if grp not in final_desc:
                continue
            for ch, texts in list(final_desc[grp].items()):
                if not texts:
                    continue

                t_list = [t for t in texts if t is not None and str(t).strip() != ""]
                if not t_list:
                    continue

                split = getattr(self, "split", "train")

                if split == "train":
                    random.shuffle(t_list)
                else:
                    # determinista por canal (y opcionalmente por split)
                    rng = random.Random(BASE_SEED + hash((split, grp, ch)) % (2**31 - 1))
                    rng.shuffle(t_list)

                final_desc[grp][ch] = t_list[: self.max_desc_per_channel]

        # ============================================================
        # 4) Descripción W/B con build_lbm_description
        # ============================================================
        #print("max_desc_per_channel",self.max_desc_per_channel)
        desc_txt = build_lbm_description(
            final_desc,
            max_per_channel=self.max_desc_per_channel,
            section_headers=True,
        )
        
        rag_txt = ""
        if self.use_rag_description:
            rag_txt = str(src.get("rag_description") or "").strip()
            if rag_txt:
                rag_txt = self.rag_prefix + rag_txt


        # ============================
        # 5) Combinación final
                # ============================
        parts = []
        if base.strip():
            parts.append(base.strip())
        if desc_txt.strip():
            parts.append(desc_txt.strip())
        if rag_txt:
            parts.append(rag_txt)

        return "\n\n".join(parts)





    # ----------------------------------------------
    # Métodos Dataset
    # ----------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        Xw = None if self._Xw is None else torch.from_numpy(self._Xw[idx])  # (Cw, d_w)
        Xb = None if self._Xb is None else torch.from_numpy(self._Xb[idx])  # (Cb, d_b)
        y  = int(self._y[idx])

        text = ""
        if self.desc_rows is not None:
            text = self._build_text(self.desc_rows[idx])

        return {
            "Xw": Xw,
            "Xb": Xb,
            "y": y,
            "text": text,
            "meta": {"idx": idx},
        }


