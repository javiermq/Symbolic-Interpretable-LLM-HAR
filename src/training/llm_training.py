import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, yaml, random, torch, numpy as np
import torch.nn as nn
from pathlib import Path

import random

from src.data.jsonl_marble import JsonlWindowsMarble  # ← tu dataset (lee window + text + y)


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True



@torch.no_grad()
def run_epoch_inference(
    injector,
    tokenizer,
    sample,
    device,
    max_new_tokens=128,
    prompt="Describe the signals briefly",
):
    """
    sample: dict con "Xw" y "Xb" (sin batch). Puede ser None en uno de ellos.
    """
    llm = injector.enc_text
    llm.eval()

    # preparar batch size = 1
    Xw = sample["Xw"]
    Xb = sample["Xb"]

    Xw = Xw.unsqueeze(0).to(device, dtype=torch.float32) if Xw is not None else None
    Xb = Xb.unsqueeze(0).to(device, dtype=torch.float32) if Xb is not None else None

    # construir inputs_embeds + attention_mask usando el injector
    inputs_embeds, attention_mask = injector.build_sequence(Xw, Xb, texts=[prompt])


    gen_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,          
        num_beams=3,
        repetition_penalty=1.05,  
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    gen_part = gen_ids[0][inputs_embeds.size(1):inputs_embeds.size(1)+50]
    print("GEN TOKENS:", tokenizer.convert_ids_to_tokens(gen_part))


    # decodificar solo lo generado
    gen_text = tokenizer.decode(
        gen_ids[0][inputs_embeds.size(1):],
        skip_special_tokens=False,
    )
    return gen_text.strip()

from pathlib import Path
import torch

def save_checkpoint_only(
    out_dir: str,
    epoch: int,
    tok_text,
    enc_text,
    feature_mlp,
    head,
    extra: dict | None = None,
):
    """
    Guarda todo lo necesario para evaluación posterior / chat:
    - tokenizer (con tokens especiales)
    - LLaMA (con vocab redimensionado)
    - projector (feature_mlp)
    - head HAR
    - metadatos (canales, clases, etc.)
    """
    ckpt_dir = Path(out_dir) / f"ep{epoch:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- tokenizer ---
    tok_text.save_pretrained(str(ckpt_dir / "tokenizer"))

    # --- LLaMA ---
    base = enc_text.base if hasattr(enc_text, "base") else enc_text
    base.save_pretrained(str(ckpt_dir / "llama"))

    # --- módulos entrenables ---
    torch.save(
        {
            "epoch": epoch,
            "feature_mlp": feature_mlp.state_dict(),
            "head": head.state_dict(),
            "extra": extra or {},
        },
        ckpt_dir / "train_state.pt",
    )

    print(f"[CKPT] saved -> {ckpt_dir}")


def debug_dtypes_step(X, Z_ch, S_ch, E_text, Z_proj, out, attn_total, h, logits,
                      prefix="[StepDebug]"):
    import torch

    def info(name, t):
        if t is None:
            print(f"{prefix} {name:12s}: None")
            return
        finite = bool(torch.isfinite(t).all().item())
        print(f"{prefix} {name:12s}: shape={tuple(t.shape)} | "
              f"dtype={t.dtype} | device={t.device} | finite={finite}")

    print("\n" + "-"*60)
    print(prefix, "RESUMEN PRIMER BATCH")
    print("-"*60)

    info("X", X)
    info("Z_ch", Z_ch)
    info("S_ch", S_ch)
    info("E_text", E_text)
    info("Z_proj", Z_proj)

    # hidden_states última capa
    if out is not None and hasattr(out, "hidden_states") and out.hidden_states:
        hs_last = out.hidden_states[-1]
    else:
        hs_last = None

    info("hs_last", hs_last)
    info("attn_total", attn_total)
    info("h_pool", h)
    info("logits", logits)

    print("-"*60 + "\n")


def debug_dtypes_global(enc_text, enc_ts, projector, head, prefix="[Global]"):
    print("\n" + "="*60)
    print(prefix, "RESUMEN DTYPE/DEVICE MODELOS")
    print("="*60)

    # LLaMA
    p_llama = next(enc_text.parameters())
    print(f"{prefix} LLaMA:      dtype={p_llama.dtype} | device={p_llama.device}")

    # Chronos
    p_chronos = next(enc_ts.enc.parameters())
    print(f"{prefix} Chronos:    dtype={p_chronos.dtype} | device={p_chronos.device}")

    # Projector
    p_proj = next(projector.parameters())
    print(f"{prefix} Projector:  dtype={p_proj.dtype} | device={p_proj.device}")

    # Head
    p_head = next(head.parameters())
    print(f"{prefix} Head:       dtype={p_head.dtype} | device={p_head.device}")

    print("="*60 + "\n")


def _assert_finite(name, t):
    if t is None: 
        return
    if not torch.isfinite(t).all():
        print(f"[NaNDetect] {name} tiene valores no finitos.",
              f"min={t.min().item() if t.numel()>0 else '∅'}",
              f"max={t.max().item() if t.numel()>0 else '∅'}",
              flush=True)
        raise RuntimeError(f"Non-finite in {name}")

def _check_labels(y, n_classes):
    if y.dtype != torch.long:
        print("⚠️ y no es LongTensor →", y.dtype)
    ymin = int(y.min().item()); ymax = int(y.max().item())
    if ymin < 0 or ymax >= n_classes:
        raise RuntimeError(f"Labels fuera de rango: min={ymin}, max={ymax}, n_classes={n_classes}")





from scipy.stats import skew, kurtosis, entropy

    

    
# src/training/load_models_clean.py (o sustituye tu load_models actual)
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from transformers import BitsAndBytesConfig
import torch

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch

def _quant_kwargs(precision: str, use_cuda: bool, device_map: str | None):
    precision = (precision or "fp16").lower()
    dm = (device_map or ("cuda:0" if use_cuda else None))

    # --- Ruta 4-bit (Recomendada) ---
    if precision in ("4bit", "int4"):
        if not use_cuda:
            raise ValueError("4-bit requiere GPU CUDA; usa fp16/fp32 en CPU.")
        return dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                # Usar bfloat16 si es compatible, si no, float16
                bnb_4bit_compute_dtype=torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            # El device_map se usa con la cuantización
            device_map=dm or "auto", 
        )

    # --- Ruta FP16/BF16/FP32 ---
    torch_dtype = (
        torch.float16 if precision == "fp16" else
        torch.bfloat16 if precision in ("bf16","bfloat16") else
        torch.float32
    )
    
    # CRÍTICO: Si no hay cuantización y estamos en CUDA, usamos FP16/BF16, NUNCA FP32
    if use_cuda and torch_dtype == torch.float32:
         # Esto evita que un modelo de 8B explote en FP32
         print("WARNING: Forzando torch.float16 en CUDA para FP32 no cuantizado.")
         torch_dtype = torch.float16
         
    out = dict(torch_dtype=torch_dtype)
    if dm is not None:
        out["device_map"] = dm
    return out


# chronos_simple.py  (puedes ponerlo en el mismo archivo)
from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM



from transformers import AutoModel, AutoTokenizer
import torch
import os

def _str2dtype(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"): return torch.bfloat16
    if name in ("fp16", "float16", "half"): return torch.float16
    if name in ("fp32", "float32", "float"): return torch.float32
    # por defecto, mejor fp16 si hay CUDA, si no fp32
    return torch.float16 if torch.cuda.is_available() else torch.float32

def _maybe_make_4bit_config(dtype_str: str):
    """
    Devuelve (qconfig, compute_dtype) si hay bitsandbytes y se pide 4bit.
    Si no hay bnb o no se pide 4bit, devuelve (None, None).
    """
    prec = (dtype_str or "").lower()
    if prec not in ("4bit", "int4", "nf4"): 
        return None, None
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        # sin bnb instalado, seguimos sin cuantización
        return None, None

    # dtype para cómputo interno en 4bit (no es el dtype de pesos)
    # usa lo que nos pasen, por defecto bf16 si disponible, si no fp16
    compute_dtype = _str2dtype("bf16")
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_enable_fp32_cpu_offload=False  # no aplica en 4bit
    )
    return qconfig, compute_dtype

def load_text_model(cfg, device):
    llama_id = cfg["model"]["llama_id"]
    inf      = cfg.get("inference", {})
    use_cuda = (device.type == "cuda")

    # --- Tokenizer (LLaMA) ---
    tok_text = AutoTokenizer.from_pretrained(llama_id)
    if tok_text.pad_token is None:
        tok_text.pad_token = tok_text.eos_token
        tok_text.pad_token_id = tok_text.eos_token_id

    # --- Opciones de mapeo/memoria ---
    llama_kwargs = dict(
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        trust_remote_code=True,
    )

    max_mem = inf.get("max_memory", None)
    if isinstance(max_mem, dict):
        max_memory = {}
        if "cuda" in max_mem: max_memory["cuda:0"] = max_mem["cuda"]
        if "cpu"  in max_mem: max_memory["cpu"]    = max_mem["cpu"]
        llama_kwargs["max_memory"] = max_memory

    device_map_text = inf.get("device_map_text", ("cuda:0" if use_cuda else "cpu"))

    llama_precision = inf.get("llama_precision", "bf16")
    qconfig_4bit, compute_dtype_4bit = _maybe_make_4bit_config(llama_precision)

    qconfig = None
    if qconfig_4bit is not None:
        qconfig = qconfig_4bit
    elif (llama_precision.lower() in ("8bit", "int8")):
        try:
            from transformers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            qconfig = None

    torch_dtype = None if qconfig is not None else _str2dtype(llama_precision)

    llm = AutoModelForCausalLM.from_pretrained(
        llama_id,
        device_map=device_map_text,
        quantization_config=qconfig,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        output_hidden_states=True,   # IMPORTANTE
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False      # LM congelado (de momento)

    D_text = llm.config.hidden_size
    
    return tok_text, llm, D_text


def check_frozen(model, name="model"):
    requires_grad = [p.requires_grad for p in model.parameters()]
    total = len(requires_grad)
    frozen = sum(not rg for rg in requires_grad)
    print(f"[Check] {name}: {frozen}/{total} params frozen "
          f"({100*frozen/total:.1f}%)")



import re, hashlib




class FuzzyInjectorLite(nn.Module):
    """
    Inyector simple:
      - X_ch: (B, C, D_feat_ch)  → MLP → h_ch: (B, C, D_txt)
      - Texto → embeddings LLaMA
      - Construye:
          [BOS,
           <ch_0>, h_ch_0,
           <ch_1>, h_ch_1,
           ...,
           texto_restante...]
    """
    def __init__(self, enc_text, tokenizer, channels, D_txt, feature_mlp):
        super().__init__()
        self.enc_text  = enc_text
        self.tokenizer = tokenizer
        self.channels  = list(channels)
        self.D_txt     = D_txt
        self.feature_mlp = feature_mlp  # ya está en float32

        for p in self.enc_text.parameters():
            p.requires_grad = False

        p0 = next(self.enc_text.parameters())
        self.model_device = p0.device
        self.model_dtype  = p0.dtype

        # Precomputar ids de tokens de canal: usamos "<loc_sleep>" etc.
        self.channel_token_strs = [f"<{ch}>" for ch in self.channels]
        self.channel_token_ids = []
        vocab = self.tokenizer.get_vocab()
        for t in self.channel_token_strs:
            if t not in vocab:
                # Debería haberse añadido antes
                print(f"⚠️ Token {t} no está en vocab; usando unk.")
                tok_id = self.tokenizer.unk_token_id
            else:
                tok_id = vocab[t]
            self.channel_token_ids.append(tok_id)

    @torch.no_grad()
    def text_to_embeds(self, texts, max_len=1024):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.model_device, non_blocking=True)
        attn_mask = enc["attention_mask"].to(self.model_device, non_blocking=True)

        emb_layer = self.enc_text.get_input_embeddings()
        E = emb_layer(input_ids)
        if E.dtype != self.model_dtype:
            E = E.to(self.model_dtype)
        return E, attn_mask

    def encode_channels(self, Xw, Xb):
        h_ch = self.feature_mlp(Xw, Xb)  # (B,C,D_txt) fp32
        return h_ch.to(self.model_device, dtype=self.model_dtype)


    def build_sequence(self, Xw, Xb, texts):
        B = Xw.size(0) if Xw is not None else Xb.size(0)
        C = len(self.channels)

        # 1) Texto → embeddings
        E_text, attn_mask = self.text_to_embeds(texts)
        Bt, T_text, D = E_text.shape
        assert D == self.D_txt

        if Bt != B:
            if Bt == 1:
                E_text = E_text.expand(B, T_text, D)
                attn_mask = attn_mask.expand(B, T_text)
            else:
                raise RuntimeError(f"Batch mismatch: B_text={Bt} vs B={B}")

        # 2) Canales → h_ch
        h_ch = self.encode_channels(Xw, Xb)  # (B,C,D_txt)

        # 3) Embeddings de tokens de canal
        emb_layer = self.enc_text.get_input_embeddings()
        w = emb_layer.weight.to(self.model_device, dtype=self.model_dtype)
        ids = torch.as_tensor(self.channel_token_ids, device=self.model_device)
        E_names = w.index_select(0, ids).unsqueeze(0).expand(B, C, D)

        # 4) Intercalar [<ch>, h_ch]
        blocks = torch.stack([E_names, h_ch], dim=2).reshape(B, 2*C, D)

        # 5) Insertar tras BOS
        E_bos  = E_text[:, :1, :]
        E_tail = E_text[:, 1:, :]
        E_total = torch.cat([E_bos, blocks, E_tail], dim=1)

        # 6) Attention mask
        A_bos  = attn_mask[:, :1]
        A_tail = attn_mask[:, 1:]
        extra  = torch.ones(B, 2*C, device=self.model_device, dtype=attn_mask.dtype)
        attn_total = torch.cat([A_bos, extra, A_tail], dim=1)

        return E_total, attn_total


    def forward(self, Xw, Xb, texts, return_attn=False):
        E_total, attn_total = self.build_sequence(Xw, Xb, texts)
        out = self.enc_text(
            inputs_embeds=E_total,
            attention_mask=attn_total,
            output_hidden_states=True
        )
        return (out, attn_total) if return_attn else out


class ResMLPBlock(nn.Module):
    """
    Pre-LN residual MLP block:
      x -> LN -> Linear(h) -> GELU -> Dropout -> Linear(D) -> Dropout -> + x
    """
    def __init__(self, D: int, hidden: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, D)
        self.drop2 = nn.Dropout(dropout)

        # init conservadora (ayuda bastante en tabular)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        r = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x + r


class ResMLPProjector(nn.Module):
    """
    D_in -> Linear(D) -> 2x ResMLPBlock(D) -> LN -> Linear(D_out)
    """
    def __init__(self, D_in: int, D: int, D_out: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(D_in, D)
        self.b1 = ResMLPBlock(D, hidden=D, dropout=dropout)     # hidden interno = D (típico ResMLP)
        self.b2 = ResMLPBlock(D, hidden=D, dropout=dropout)
        self.ln_out = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, D_out)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.ln_out(x)
        x = self.out_proj(x)
        return x


class FeatureMLPByType(nn.Module):
    """
    Entrada:
      Xw: (B, Cw, d_w)  | None
      Xb: (B, Cb, d_b)  | None
    Salida:
      H: (B, C, D_txt) en el ORDEN GLOBAL ds_tr.channels
    """
    def __init__(self, channels, wear_cols, bin_cols, d_w, d_b, D_txt,
                 hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.channels  = list(channels)
        self.wear_cols = list(wear_cols)
        self.bin_cols  = list(bin_cols)
        self.d_w = int(d_w) if d_w is not None else None
        self.d_b = int(d_b) if d_b is not None else None
        self.D_txt = int(D_txt)

        # posiciones globales
        pos = {ch:i for i,ch in enumerate(self.channels)}
        self.wear_pos = [pos[ch] for ch in self.wear_cols]
        self.bin_pos  = [pos[ch] for ch in self.bin_cols]

        # MLP por tipo
        self.wear_mlp = ResMLPProjector(D_in=self.d_w, D=hidden, D_out=self.D_txt, dropout=dropout) if self.d_w else None
        self.bin_mlp  = ResMLPProjector(D_in=self.d_b, D=hidden, D_out=self.D_txt, dropout=dropout) if self.d_b else None

    def forward(self, Xw: torch.Tensor | None, Xb: torch.Tensor | None) -> torch.Tensor:
        if Xw is None and Xb is None:
            raise RuntimeError("FeatureMLPByType.forward: Xw y Xb son None")

        base = Xw if Xw is not None else Xb
        B = base.size(0)
        C = len(self.channels)

        out = torch.zeros((B, C, self.D_txt), device=base.device, dtype=torch.float32)

        if Xw is not None and self.wear_mlp is not None:
            Bw, Cw, dw = Xw.shape
            assert dw == self.d_w
            Hw = self.wear_mlp(Xw.reshape(Bw * Cw, dw)).reshape(Bw, Cw, self.D_txt)
            out[:, self.wear_pos, :] = Hw

        if Xb is not None and self.bin_mlp is not None:
            Bb, Cb, db = Xb.shape
            assert db == self.d_b
            Hb = self.bin_mlp(Xb.reshape(Bb * Cb, db)).reshape(Bb, Cb, self.D_txt)
            out[:, self.bin_pos, :] = Hb

        return out




class FeatureMLP(nn.Module):
    def __init__(self, D_in, D_hidden, D_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_in, D_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_hidden, D_out),
        )

    def forward(self, x):
        return self.net(x)
        





# --------------------------
# 3) Pool de las 2 últimas capas (SensorLLM-style) + clasificador MLP
# --------------------------
class Last2LayerPool(nn.Module):
    """
    Pool estilo SensorLLM: toma las 2 últimas capas,
    hace mean-pool sobre tokens y luego promedio de capas.
    """
    def __init__(self): super().__init__()
    def forward(self, hidden_states):
        # hidden_states: tuple(len=L+1) incluyendo capa 0 de embeddings → quitamos la 0
        hs = hidden_states[1:]
        h1, h2 = hs[-2], hs[-1]                 # (B, T, D)
        p1 = h1.mean(dim=1)                     # (B, D)
        p2 = h2.mean(dim=1)                     # (B, D)
        return 0.5*(p1+p2)                      # (B, D)

class MLPHead(nn.Module):
    def __init__(self, D_txt, hidden=1024, n_classes=10, dropout=0.1, act="gelu"):
        super().__init__()

        # El head SIEMPRE en float32, aunque LLaMA esté en 4bit/fp16
        self.net = nn.Sequential(
            nn.Linear(D_txt, hidden, dtype=torch.float32),
            nn.GELU() if act=="gelu" else nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes, dtype=torch.float32),
        )

        # Inicialización estable
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Asegura entrada en FP32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.net(x)


import torch
import torch.nn as nn


def sanitize_module(module, name="module"):
    """
    Revisa todos los parámetros de `module` y sustituye NaN/Inf
    por valores finitos.
    """
    any_bad = False
    with torch.no_grad():
        for pname, p in module.named_parameters():
            if p is None:
                continue

            # Usar torch.isfinite (CORRECTO en PyTorch >=1.10,2.x)
            mask = ~torch.isfinite(p)

            if mask.any():
                any_bad = True
                print(f"[Sanitize] {name}.{pname} tiene {mask.sum().item()} valores no finitos.")
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)

    if any_bad:
        print(f"[Sanitize] Parámetros no finitos detectados y corregidos en {name}.")

def sanitize_head(head, name="head"):
    any_bad = False
    with torch.no_grad():
        for pname, p in head.named_parameters():
            if p is None:
                continue

            # SOLO mirar p.data (no grad)
            mask = ~torch.isfinite(p.data)
            if mask.any():
                any_bad = True
                print(f"[Sanitize] {name}.{pname} PESOS no finitos ({mask.sum().item()})")
    if any_bad:
        print("[Sanitize] ¡PESOS no finitos detectados!")

        
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _print_baseline_metrics(name, y_train, y_test, y_tr_pred, y_te_pred):
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




# --------------------------
# 4) Entrenamiento (fase única combinada): projector + clasificador
# --------------------------
# --------------------------
# 4) Entrenamiento (fase única fuzzy+stats → MLP → LLaMA)
# --------------------------
from sklearn.metrics import f1_score
from src.data.jsonl_marble_fuzzy_stats import JsonlMarbleFuzzyStats, set_channel_type_map

def train_once(cfg_path: str, device: str, cfg2_path: str | None = None):
    import yaml
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import numpy as np

    from src.data.jsonl_marble_fuzzy_stats import JsonlMarbleFuzzyStats

    cfg = yaml.safe_load(open(cfg_path, "r"))
    data_cfg = cfg["data"]

    # ================================================
    # Cargar config2 (marble.cfg) para actividades y canales
    # ================================================
    if cfg2_path is not None:
        print(f"[INFO] Cargando configuración secundaria (marble): {cfg2_path}")
        cfg2 = yaml.safe_load(open(cfg2_path, "r"))

        # 1) Actividades
        acts = cfg2.get("activities", None)
        if acts is not None:
            print(f"[CONFIG2] activities desde marble.cfg ({len(acts)})")
            data_cfg["classes"] = acts

        # 2) Canales: sensors.wear + sensors.binary
        sensors_cfg = cfg2.get("sensors", {}) or {}
        wear_ch   = [str(x).strip() for x in (sensors_cfg.get("wear", []) or [])]
        binary_ch = [str(x).strip() for x in (sensors_cfg.get("binary", []) or [])]
        channels2 = wear_ch + binary_ch
        if channels2:
            print(f"[CONFIG2] {len(channels2)} canales desde marble.cfg (wear+binary)")
            data_cfg["channels"] = channels2

        # 3) Mapa canal → tipo (WEAR/BINARY) para fuzzy+stats
        channel_type_map = {}
        for ch in wear_ch:
            channel_type_map[ch] = "WEAR"
        for ch in binary_ch:
            if ch in channel_type_map and channel_type_map[ch] != "BINARY":
                print(f"[CONFIG2][WARN] Canal '{ch}' duplicado como WEAR y BINARY; "
                      f"me quedo con BINARY.")
            channel_type_map[ch] = "BINARY"

        # Inyectar en jsonl_marble_fuzzy_stats → rellenará WEAR_SENSORS/BINARY_SENSORS
        set_channel_type_map(channel_type_map)


    label_field = data_cfg.get("label_field", "label")
    print("label_field", label_field)

    activities_cfg = data_cfg.get("classes")
    print("activities_cfg", activities_cfg)


    # JsonlMarbleFuzzyStats construye lab2id a partir de los labels del JSONL
    activities = None

    device = torch.device(device)

    # ======================
    # 1) Dataset fuzzy+stats
    # ======================
    use_rag = bool(data_cfg.get("use_rag_description", False))
    print("USE RAG?",use_rag)
    
    ds_tr = JsonlMarbleFuzzyStats(
        raw_path=data_cfg["train_raw"],
        desc_path=data_cfg["train_desc"],   # 👈 nuevo
        features_path=data_cfg["train_features"],
        channels=data_cfg["channels"],
        fs=data_cfg["fs"],
        duration=data_cfg["duration"],
        text_base=data_cfg.get("text_base", ""),
        max_desc_per_channel=int(data_cfg.get("max_desc_per_channel", 4)),
        activities=activities_cfg,
        use_rag_description=use_rag,
        rag_prefix=data_cfg.get("rag_prefix", "Answer help: "),
    )
    
    print("\n[DEBUG] Primera fila desc_rows (train):")
    if ds_tr.desc_rows is None:
        print("desc_rows es None (NO se ha cargado ningún fichero de descripciones)")
    else:
        print(type(ds_tr.desc_rows[0]), ds_tr.desc_rows[0])

    ds_va = JsonlMarbleFuzzyStats(
        raw_path=data_cfg["val_raw"],
        desc_path=data_cfg["val_desc"],     # 👈 nuevo
        features_path=data_cfg["val_features"],
        channels=data_cfg["channels"],
        fs=data_cfg["fs"],
        duration=data_cfg["duration"],
        text_base=data_cfg.get("text_base", ""),
        max_desc_per_channel=int(data_cfg.get("max_desc_per_channel", 4)),
        activities=activities_cfg,
        use_rag_description=use_rag,
        rag_prefix=data_cfg.get("rag_prefix", "Answer help: "),
    )


    n_classes = len(ds_tr.lab2id)
    print("n_classes", n_classes)
    print("lab2id:", ds_tr.lab2id)

    # Dimensión de features (fuzzy+stats)
    
    C = len(ds_tr.channels)
    d_w = ds_tr.d_w
    d_b = ds_tr.d_b
    print("C=", C, "d_w=", d_w, "d_b=", d_b)


    # tipos desde NPZ
    ch_types = getattr(ds_tr, "channel_types", None)
    if ch_types is None:
        raise RuntimeError("Dataset no expone channel_types. Añade self.channel_types en jsonl_marble_fuzzy_stats.py")

    wear_pos   = [i for i,t in enumerate(ch_types) if str(t).upper() == "WEAR"]
    binary_pos = [i for i,t in enumerate(ch_types) if str(t).upper() == "BINARY"]

    



    # ======================
    # 2) Pesos de clase
    # ======================
    class_counts = torch.zeros(n_classes, dtype=torch.float32)
    for i in range(len(ds_tr)):
        y_i = ds_tr[i]["y"]
        class_counts[y_i] += 1

    print("class_counts:", class_counts.tolist())

    class_weights = 1.0 / class_counts.float().clamp_min(1)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print("class_weights:", class_weights.tolist())


    # Dimensión de features (fuzzy+stats)
    # Dimensión de features (fuzzy+stats)
   
    C = len(ds_tr.channels)
    print("C canales =", C)

    
    # ======================
    # 3) Modelos base: SOLO LLaMA
    # ======================
    tok_text, enc_text, D_txt = load_text_model(cfg, device)

    # ======================
    # 3bis) Añadir tokens de canal al tokenizer
    # ======================
    vocab = tok_text.get_vocab()

    # --- tokens de canal ---
    channel_tokens = [f"<{c}>" for c in ds_tr.channels]
    new_channel_tokens = [t for t in channel_tokens if t not in vocab]


    # --- añadirlos ---
    #all_new_tokens = new_channel_tokens + new_activity_tokens
    all_new_tokens = new_channel_tokens

    if all_new_tokens:
        print(f"Añadiendo {len(all_new_tokens)} tokens simbólicos al tokenizer:")
        for t in all_new_tokens:
            print("  ", t)

        tok_text.add_special_tokens({"additional_special_tokens": all_new_tokens})
        enc_text.resize_token_embeddings(len(tok_text), mean_resizing=False)
        
        # =========================================================
        # Hacer que SOLO los tokens nuevos (canales + actividades)
        # se aprendan en el embedding space de LLaMA
        # =========================================================
        emb = enc_text.get_input_embeddings()          # nn.Embedding
        emb.weight.requires_grad = True               # activamos gradiente SOLO aquí

        # ids de tokens nuevos
        new_token_ids = [tok_text.convert_tokens_to_ids(t) for t in all_new_tokens]
        new_token_ids = [i for i in new_token_ids if i is not None and i >= 0]

        # máscara: 1 en filas de tokens nuevos, 0 en el resto
        mask = torch.zeros_like(emb.weight, dtype=torch.float32)
        mask[new_token_ids, :] = 1.0
        mask = mask.to(device=emb.weight.device, dtype=emb.weight.dtype)

        # hook para anular gradientes del vocab antiguo
        def _mask_grad(grad):
            return grad * mask

        emb.weight.register_hook(_mask_grad)
        
    else:
        print("Todos los tokens simbólicos ya estaban en el vocabulario.")


    # ignoramos enc_ts (Chronos), no lo usamos

    check_frozen(enc_text, "enc_text")

    print("CUDA available:", torch.cuda.is_available())
    print("enc_text param device:", next(enc_text.parameters()).device)
    print("enc_text dtype:", next(enc_text.parameters()).dtype)
    dm = getattr(enc_text, "hf_device_map", None)
    print("hf_device_map:", dm)

    # Dispositivo y dtype "reales" del modelo
    llama_param = next(enc_text.parameters())
    model_device = llama_param.device
    model_dtype  = llama_param.dtype

    # ======================
    # 4) Módulos entrenables: MLP(feats) + pool + head
    # ======================
    D_hidden_feats = int(cfg["model"].get("feat_mlp_hidden", 1024))
    feat_dropout   = float(cfg["model"].get("feat_mlp_dropout", 0.1))



    feature_mlp = FeatureMLPByType(
        channels=ds_tr.channels,
        wear_cols=ds_tr.wear_cols,
        bin_cols=ds_tr.bin_cols,
        d_w=d_w,
        d_b=d_b,
        D_txt=D_txt,
        hidden=D_hidden_feats,
        dropout=feat_dropout,
    ).to(model_device, dtype=torch.float32)

    injector = FuzzyInjectorLite(
        enc_text=enc_text,
        tokenizer=tok_text,
        channels=ds_tr.channels,
        D_txt=D_txt,
        feature_mlp=feature_mlp,
    ).to(model_device)


    
    pooler = Last2LayerPool().to(model_device)

    head = MLPHead(
        D_txt=D_txt,
        hidden=int(cfg["model"].get("head_hidden", 1024)),
        n_classes=n_classes,
        dropout=float(cfg["model"].get("head_dropout", 0.1)),
        act=cfg["model"].get("head_act", "gelu"),
    ).to(model_device)

    # Parámetros a entrenar: MLP + head (LLaMA congelado, inyector sólo encapsula)
    # ======================
    # Optimizador
    # ======================
    lr = float(cfg["train"].get("lr", 1e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-2))

    emb = enc_text.get_input_embeddings()

    params = (
        list(feature_mlp.parameters())
        + list(pooler.parameters())     # 👈 AÑADE ESTO
        + list(head.parameters())
        + [enc_text.get_input_embeddings().weight]
    )

    





    lr = float(cfg["train"].get("lr", 1e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-2))

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(model_device))

    # ======================
    # 5) DataLoaders (collate fuzzy+stats)
    # ======================
    
    C = len(ds_tr.channels)
    print("C canales =", C)
    


    
    def collate_fuzzy_stats(batch):
        ys    = torch.tensor([int(b["y"]) for b in batch], dtype=torch.long)
        texts = [b.get("text", "") for b in batch]
        metas = [b.get("meta", {}) for b in batch]

        # Xw/Xb pueden ser None si no hay ese tipo
        Xw_list = [b["Xw"] for b in batch]
        Xb_list = [b["Xb"] for b in batch]

        Xw = None if Xw_list[0] is None else torch.stack(Xw_list, dim=0)  # (B,Cw,dw)
        Xb = None if Xb_list[0] is None else torch.stack(Xb_list, dim=0)  # (B,Cb,db)

        return {"Xw": Xw, "Xb": Xb, "y": ys, "text": texts, "meta": metas}



    B = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 4))

    dl_tr = DataLoader(
        ds_tr,
        batch_size=B,
        shuffle=True,
        collate_fn=collate_fuzzy_stats,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=B,
        shuffle=False,
        collate_fn=collate_fuzzy_stats,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    
    
    # ======================
    # Baseline SVM (robusto a Xw=None / Xb=None)
    # ======================
    def _flat_npz(ds):
        parts = []
        if getattr(ds, "_Xw", None) is not None:
            parts.append(ds._Xw.reshape(len(ds), -1))
        if getattr(ds, "_Xb", None) is not None:
            parts.append(ds._Xb.reshape(len(ds), -1))
        if not parts:
            raise RuntimeError("No hay features: ds._Xw y ds._Xb son None")
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    X_train = _flat_npz(ds_tr)
    X_test  = _flat_npz(ds_va)

    y_train = np.array([ds_tr[i]["y"] for i in range(len(ds_tr))])
    y_test  = np.array([ds_va[i]["y"] for i in range(len(ds_va))])


    C = len(ds_tr.channels)


    print(tok_text.tokenize("<"+ds_tr.channels[0]+">"))
    print(tok_text.convert_tokens_to_ids("<"+ds_tr.channels[0]+">"))

    
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
#    from sklearn.svm import LinearSVC

#    svm = Pipeline([
#        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
#        ("clf", LinearSVC(C=1.0))
#    ])

#    svm.fit(X_train, y_train)

#    y_svm_tr = svm.predict(X_train)
#    y_svm_te = svm.predict(X_test)

#    _print_baseline_metrics("SVM (fuzzy+stats)", y_train,y_test, y_svm_tr, y_svm_te)


    # ======================
    # 6) Bucle de entrenamiento
    # ======================
    import time

    epochs = int(cfg["train"]["epochs"])

    lambda_lm = float(cfg["train"].get("lambda_lm", 1.0))
    lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        
        
    for ep in range(1, epochs + 1):
        enc_text.eval()          # congelado
        feature_mlp.train()
        head.train()

        t0 = time.time()
        tot_loss = 0.0
        tot_acc = 0.0
        total_lm = 0.0
        tot_n = 0
        tot_loss_cls = 0.0

        all_labels = []
        all_preds = []

        n_batches = len(dl_tr)
        
        if len(ds_tr) > 0:
            rnd_i = random.randint(0, len(ds_tr) - 1)
            rnd_row = ds_tr[rnd_i]

            y_raw = rnd_row["y"]
            # soporta tanto int como tensor escalar por si cambias el dataset en el futuro
            if isinstance(y_raw, torch.Tensor):
                y_idx = int(y_raw.item())
            else:
                y_idx = int(y_raw)

            sample_label = ds_tr.id2lab[y_idx]

            print(f"\n🟢 [Ejemplo epoch {ep}] label: {sample_label}")
            text_preview = rnd_row.get("text", "") or ""
            text_preview = str(text_preview).replace("\n", " ")
            print(f"Texto (inicio): {text_preview}...\n")

            Xw_r = rnd_row["Xw"]
            Xb_r = rnd_row["Xb"]

            print("Xw shape:", None if Xw_r is None else tuple(Xw_r.shape))
            print("Xb shape:", None if Xb_r is None else tuple(Xb_r.shape))

            if Xw_r is not None:
                vals = Xw_r.flatten().tolist()
                print("Xw first vals:", ", ".join(f"{v:.2f}" for v in vals))
            if Xb_r is not None:
                vals = Xb_r.flatten().tolist()
                print("Xb first vals:", ", ".join(f"{v:.2f}" for v in vals))



        for b, batch in enumerate(dl_tr, start=1):
            Xw = batch["Xw"].to(model_device, dtype=torch.float32) if batch["Xw"] is not None else None
            Xb = batch["Xb"].to(model_device, dtype=torch.float32) if batch["Xb"] is not None else None
            y      = batch["y"].to(model_device)
            texts  = batch["text"]

            # --- Forward con inyector + LLaMA causal ---
            out, attn_total = injector(Xw, Xb, texts, return_attn=True)
            hidden_states = out.hidden_states      # para HAR
            lm_logits     = out.logits             # (B, T_total, Vocab)

            # --- HAR: pool + head ---
            h = pooler(hidden_states)              # (B, D_txt)
            _assert_finite("h (train)", h)

            logits_cls = head(h)                   # (B, n_classes)
            _assert_finite("logits (train)", logits_cls)
            loss_cls   = criterion(logits_cls.float(), y)

            # ==========================
            # LM: reconstruir texto LBM
            # ==========================
            enc = tok_text(
                texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            input_ids_text = enc["input_ids"].to(model_device)          # (B, T_text)
            attn_text      = enc["attention_mask"].to(model_device)     # (B, T_text)

            B_batch, T_text = input_ids_text.shape
            C = len(ds_tr.channels)

            T_total = lm_logits.size(1)

            lm_labels = torch.full(
                (B_batch, T_total),
                fill_value=-100,
                dtype=torch.long,
                device=model_device,
            )

            start = 1 + 2 * C
            T_tail = T_text - 1
            end = min(start + T_tail, T_total)
            T_effective = end - start

            if T_effective > 0:
                lm_labels[:, start:end] = input_ids_text[:, 1:1+T_effective]

                # ✅ Ignorar PAD dentro del tramo supervisado
                tail_mask = attn_text[:, 1:1+T_effective]          # 1 si real, 0 si PAD
                lm_labels[:, start:end][tail_mask == 0] = -100

            # ✅ Shift causal (next-token)
            V = lm_logits.size(-1)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()

            loss_lm = lm_criterion(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
            )


            # --- Loss total ---
            loss = loss_cls + lambda_lm * loss_lm

            # --- Métricas HAR ---
            preds = logits_cls.argmax(dim=-1)
            acc = (preds == y).float().mean().item()

            # --- Backward + optim ---
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # --- Acumuladores ---
            tot_loss     += loss.item() * y.size(0)
            tot_loss_cls += loss_cls.item() * y.size(0)
            total_lm     += loss_lm.item() * y.size(0)
            tot_acc      += acc * y.size(0)
            tot_n        += y.size(0)

            all_labels.extend(y.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

            progress = 100.0 * b / n_batches
            avg_loss = tot_loss / max(tot_n, 1)
            avg_cls  = tot_loss_cls / max(tot_n, 1)
            avg_lm   = total_lm / max(tot_n, 1)
            avg_lmw  = lambda_lm * avg_lm
            avg_acc  = tot_acc / max(tot_n, 1)


            bar = (
                f"\r[Epoch {ep}] {progress:6.2f}% | "
                f"batch {b:4d}/{n_batches:<4d} | "
                f"loss={avg_loss:.4f} | "
                f"cls={avg_cls:.4f} | "
                f"lm={avg_lm:.3f} | "
                f"λlm={avg_lmw:.3f} | "
                f"acc={avg_acc:.3f}"
            )
            import sys
            sys.stdout.write(bar)
            sys.stdout.flush()


        dt = time.time() - t0
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        print(
            f"\nEpoch {ep} completed in {dt:.1f}s | "
            f"loss={avg_loss:.4f} | "
            f"cls={avg_cls:.4f} | "
            f"lm={avg_lm:.3f} | "
            f"λlm={avg_lmw:.3f} | "
            f"acc={avg_acc:.3f} | "
            f"f1-macro={f1_macro:.3f} | f1-weighted={f1_weighted:.3f}"
        )

        # 7) Validación
        # ======================

        enc_text.eval()
        feature_mlp.eval()
        head.eval()

        tot_loss_va    = 0.0   # loss total (HAR + λ·LM)
        tot_loss_cls_va = 0.0  # solo HAR
        tot_loss_lm_va  = 0.0  # solo LM
        tot_acc_va     = 0.0
        tot_n_va       = 0
        all_labels_va  = []
        all_preds_va   = []

        C = len(ds_tr.channels)  # mismos canales que en train

        with torch.no_grad():
            for batch in dl_va:
                Xw = batch["Xw"].to(model_device, dtype=torch.float32) if batch["Xw"] is not None else None
                Xb = batch["Xb"].to(model_device, dtype=torch.float32) if batch["Xb"] is not None else None
                y      = batch["y"].to(model_device)
                texts  = batch["text"]

                # --- Forward con inyector + LLaMA causal ---
                out, attn_total = injector(Xw, Xb, texts, return_attn=True)
                hidden_states = out.hidden_states
                lm_logits     = out.logits             # (B, T_total, V)

                # --- HAR ---
                h = pooler(hidden_states)
                logits_cls = head(h)
                loss_cls   = criterion(logits_cls.float(), y)

                # --- LM: reconstruir texto LBM (igual que en train, pero sin backward) ---
                enc = tok_text(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                input_ids_text = enc["input_ids"].to(model_device)
                attn_text      = enc["attention_mask"].to(model_device)
                
                B_batch, T_text = input_ids_text.shape

                T_total = lm_logits.size(1)

                lm_labels = torch.full(
                    (B_batch, T_total),
                    fill_value=-100,
                    dtype=torch.long,
                    device=model_device,
                )

                start = 1 + 2 * C          # donde empieza el texto en la secuencia
                T_tail = T_text - 1        # texto sin BOS
                end = start + T_tail
                end = min(end, T_total)
                T_effective = end - start

                if T_effective > 0:
                    lm_labels[:, start:end] = input_ids_text[:, 1:1+T_effective]

                tail_mask = attn_text[:, 1:1+T_effective]
                lm_labels[:, start:end][tail_mask == 0] = -100

                V = lm_logits.size(-1)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = lm_labels[:, 1:].contiguous()

                loss_lm = lm_criterion(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                )

                # --- Loss total de validación ---
                loss = loss_cls + lambda_lm * loss_lm

                # --- Métricas HAR ---
                preds = logits_cls.argmax(dim=-1)
                acc = (preds == y).float().mean().item()

                tot_loss_va     += loss.item() * y.size(0)
                tot_loss_cls_va += loss_cls.item() * y.size(0)
                tot_loss_lm_va  += loss_lm.item() * y.size(0)
                tot_acc_va      += acc * y.size(0)
                tot_n_va        += y.size(0)

                all_labels_va.extend(y.detach().cpu().tolist())
                all_preds_va .extend(preds.detach().cpu().tolist())

        avg_loss_va     = tot_loss_va     / max(tot_n_va, 1)
        avg_loss_cls_va = tot_loss_cls_va / max(tot_n_va, 1)
        avg_loss_lm_va  = tot_loss_lm_va  / max(tot_n_va, 1)
        avg_acc_va      = tot_acc_va      / max(tot_n_va, 1)
        f1_macro_va     = f1_score(all_labels_va, all_preds_va, average="macro")
        f1_weighted_va  = f1_score(all_labels_va, all_preds_va, average="weighted")

        print(
            f"[VAL] Epoch {ep} | "
            f"loss_tot={avg_loss_va:.4f} | "
            f"loss_cls={avg_loss_cls_va:.4f} | "
            f"loss_lm={avg_loss_lm_va:.4f} | "
            f"acc={avg_acc_va:.3f} | "
            f"f1-macro={f1_macro_va:.3f} | f1-weighted={f1_weighted_va:.3f}"
        )

        # ---------- INFERENCE DEBUG ----------
        print("\n[Inference check]")

        idx_tr = random.randint(0, len(ds_tr) - 1)
        sample_tr = ds_tr[idx_tr]
        text_tr = run_epoch_inference(
            injector=injector,
            tokenizer=tok_text,
            sample=sample_tr,
            device=model_device,
        )
        print(f"[TRAIN #{idx_tr}] {text_tr}")

        idx_va = random.randint(0, len(ds_va) - 1)
        sample_va = ds_va[idx_va]
        text_va = run_epoch_inference(
            injector=injector,
            tokenizer=tok_text,
            sample=sample_va,
            device=model_device,
        )
        print(f"[VAL #{idx_va}] {text_va}")
        print("-" * 80)
        
        if ep % 1 == 0:
            save_checkpoint_only(
                out_dir=cfg["train"].get("out_dir", "outputs_clean"),
                epoch=ep,
                tok_text=tok_text,
                enc_text=enc_text,
                feature_mlp=feature_mlp,
                head=head,
                extra={
                    "channels": ds_tr.channels,
                    "lab2id": ds_tr.lab2id,
                    "id2lab": ds_tr.id2lab,
                    "lambda_lm": lambda_lm,
                },
            )



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--config2", type=str, required=False, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    train_once(args.config, args.device, cfg2_path=args.config2)
