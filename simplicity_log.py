"""
Chronos Simplicity Bias Suite v3 (Fourier target bits)
======================================================

Enhancements over v2:
- For --generator fourier, you can request many *code-length* points by
  specifying a target bits grid, e.g. --fourier_bits_grid 10:100:10
  (or a comma list like 10,20,30,...). The generator picks K (number of
  Fourier components) that yields a code length closest to each target,
  giving you dense complexity points.
- Adaptive plotting bins: baseline plot now uses up to min(unique K_bits, 24)
  bins so all classes show up (use --plot_bins to override).

Other features are unchanged (Occam pairs, calibration, iso-difficulty, token
monotone acceptance).

"""

import os
import re
import glob
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSeq2SeqLM
import pandas as pd
import matplotlib.pyplot as plt

# Progress bars ---------------------------------------------------------------
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # fallback: no-op
        return x



# -------------------------------------------------------------------------
# Robust import of `chronos` (installed or local "chronos*.py" file)
# -------------------------------------------------------------------------
try:
    import chronos  # type: ignore
except Exception:
    here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = sorted(glob.glob(os.path.join(here, "chronos*.py")))
    if not candidates:
        raise ImportError(
            "Could not import `chronos` and no local 'chronos*.py' file found.\n"
            "Place your Chronos implementation (e.g., 'chronos (5).py') in the same folder as this script."
        )
    chronos_path = candidates[-1]
    import importlib.util
    spec = importlib.util.spec_from_file_location("chronos", chronos_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Chronos module from {chronos_path}")
    chronos = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chronos)  # type: ignore

if not hasattr(chronos, "ChronosConfig"):
    raise ImportError("The loaded `chronos` module has no `ChronosConfig`.")

ChronosConfig = chronos.ChronosConfig  # type: ignore

# ------------------------------
# Model IDs
# ------------------------------
MODEL_ID_MAP = {
    "tiny":  "amazon/chronos-t5-tiny",
    "mini":  "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base":  "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}

# ------------------------------
# Data containers
# ------------------------------

@dataclass
class SeriesSpec:
    y: np.ndarray
    K_bits: float
    family: str
    params: Dict[str, float]
    sid: int

@dataclass
class OccamPair:
    sid: int
    family: str
    K_simple: float
    K_complex: float
    delta_K: float
    y_context: np.ndarray          # length C
    y_simple_future: np.ndarray    # length H_max
    y_complex_future: np.ndarray   # length H_max

# ------------------------------
# Helpers: bits & z-score
# ------------------------------

def _bits_for_choice(n: int) -> float:
    return math.ceil(math.log2(max(n, 1)))

def _bits_for_int_range(low: int, high: int) -> float:
    assert high >= low
    return _bits_for_choice(high - low + 1)

def _bits_for_real_grid(n: int) -> float:
    return _bits_for_choice(n)

def _zscore(y: np.ndarray) -> np.ndarray:
    mu = float(np.mean(y)); sd = float(np.std(y) + 1e-6)
    return (y - mu) / sd

# ------------------------------
# Chronos-aware tokenizer & adapter
# ------------------------------

class ChronosTokenizerAdapter:
    def __init__(self, hf_config):
        assert hasattr(hf_config, "chronos_config"), "Not a Chronos config (missing chronos_config)."
        self.chronos_cfg = ChronosConfig(**hf_config.chronos_config)
        self.tokenizer = self.chronos_cfg.create_tokenizer()
        self.pred_len = self.chronos_cfg.prediction_length
        self.use_eos = getattr(self.chronos_cfg, "use_eos_token", False)

    def tokenize_context(self, y_context: np.ndarray) -> Tuple[np.ndarray, np.ndarray, object]:
        ctx = torch.tensor(y_context, dtype=torch.float32).unsqueeze(0)
        ctx_ids, ctx_mask, state = self.tokenizer.context_input_transform(ctx)
        return (ctx_ids[0].cpu().numpy(),
                ctx_mask[0].cpu().numpy().astype(np.int64),
                state)

    def tokenize_given_target(self, y_target: np.ndarray, state) -> Tuple[np.ndarray, np.ndarray]:
        tgt = torch.tensor(y_target, dtype=torch.float32).unsqueeze(0)
        lbl_ids, lbl_mask = self.tokenizer.label_input_transform(tgt, state)
        return (lbl_ids[0].cpu().numpy(),
                lbl_mask[0].cpu().numpy().astype(np.int64))

class ChronosAdapter:
    def __init__(self, model_id: str, device: str = "cuda", dtype: Optional[torch.dtype] = None):
        self.model_id = model_id
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        self.config = AutoConfig.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
        self.model.eval()

        self.pad_token_id = getattr(self.config, "pad_token_id", 0)
        self.decoder_start_token_id = getattr(self.config, "decoder_start_token_id", self.pad_token_id)
        self.vocab_size = getattr(self.config, "vocab_size", 4096)

        self.ctok = ChronosTokenizerAdapter(self.config)

    @torch.no_grad()
    def score_batch(self, context_ids: torch.LongTensor, target_ids: torch.LongTensor,
                    attention_mask: Optional[torch.LongTensor] = None,
                    temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            input_ids=context_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            use_cache=False,
        )
        logits = out.logits  # [B, H, V]
        if temperature is None or temperature <= 0:
            temperature = 1.0
        logits = logits / float(temperature)

        B, H, V = logits.shape
        logp = torch.log_softmax(logits.float(), dim=-1)
        labels = target_ids.view(B, H, 1)
        tok_logp = torch.gather(logp, dim=-1, index=labels).squeeze(-1)  # [B, H]

        if self.pad_token_id is not None:
            valid = (target_ids != self.pad_token_id).to(tok_logp.dtype)
        else:
            valid = torch.ones_like(tok_logp)

        tok_logp = tok_logp * valid
        lengths = valid.sum(dim=1).clamp(min=1.0)
        total_logprob = tok_logp.sum(dim=1)
        token_nll = -total_logprob / lengths
        return token_nll, total_logprob

# ------------------------------
# Token complexity metrics
# ------------------------------

def runs_complexity(ids: np.ndarray) -> float:
    if ids.size == 0: return 0.0
    return float(1 + np.count_nonzero(ids[1:] != ids[:-1]))

def lz_complexity(ids: np.ndarray) -> float:
    s = [int(x) for x in ids.tolist()]
    i = 0; k = 1; l = 1; c = 1; n = len(s)
    while True:
        if i + k > n - 1:
            c += 1; break
        if s[i+k] == s[l+k]:
            k += 1
            if l + k > n - 1:
                c += 1; break
        else:
            if k > 0:
                i += 1
            else:
                c += 1; l += 1; i = 0
            k = 1
            if l > n - 1:
                break
    return float(c)

# ------------------------------
# Fourier generator with target-bits
# ------------------------------

def _fourier_code_bits(K: int, F_size: int, bits_A: int, bits_phi: int, n_classes: int) -> float:
    if K < 0 or K > F_size: return float('inf')
    # choose class id (K) + choose K freqs + params per component
    comb_bits = _bits_for_choice(math.comb(F_size, K)) if 0 < K < F_size else 0.0
    return _bits_for_choice(n_classes) + comb_bits + K*(bits_A + bits_phi)

def parse_bits_grid(bits_grid: str) -> List[float]:
    bits_grid = bits_grid.strip()
    if not bits_grid:
        return []
    if ':' in bits_grid:
        # format: start:stop:step
        start, stop, step = bits_grid.split(':')
        start = float(start); stop = float(stop); step = float(step)
        # inclusive of stop if exactly divisible
        vals = []
        x = start
        while x <= stop + 1e-9:
            vals.append(round(x, 6))
            x += step
        return vals
    else:
        return [float(x) for x in bits_grid.split(',') if x.strip()]

def gen_fourier_suite(num_series: int, T: int, K_list: List[int],
                      seed: int = 0, F_size: int = 64) -> List[SeriesSpec]:
    rng = np.random.default_rng(seed)
    out: List[SeriesSpec] = []
    F = np.linspace(1/T, 0.45, F_size)  # candidate freqs
    A_grid = np.linspace(0.3, 1.8, 16)
    phi_grid = np.linspace(0, 2*np.pi, 32, endpoint=False)

    bits_F = _bits_for_real_grid(len(F))
    bits_A = _bits_for_real_grid(len(A_grid))
    bits_phi = _bits_for_real_grid(len(phi_grid))
    n_classes = max(1, len(set(K_list)))
    sid = 0
    per_class = max(1, num_series // n_classes)
    for K in K_list:
        # precompute code length (constant per K)
        K_bits = _fourier_code_bits(K, len(F), bits_A, bits_phi, n_classes)
        for _ in range(per_class):
            # choose K well-separated freqs
            freqs = []
            while len(freqs) < K:
                f = float(rng.choice(F))
                ok = True
                for g in freqs:
                    if abs(f - g) < 2*(F[1]-F[0]):
                        ok = False; break
                if ok: freqs.append(f)
            amps = [float(rng.choice(A_grid)) for _ in range(K)]
            phis = [float(rng.choice(phi_grid)) for _ in range(K)]
            t = np.arange(T, dtype=np.float32)
            y = np.zeros(T, dtype=np.float32)
            for a, f, p in zip(amps, freqs, phis):
                y += a*np.sin(2*np.pi*f*t + p)
            y = _zscore(y)
            out.append(SeriesSpec(y=y, K_bits=float(K_bits), family=f"FourierK{K}",
                                  params={"K": K}, sid=sid)); sid += 1
    return out

def pick_K_for_bits_targets(bits_targets: List[float], T: int, F_size: int,
                            bits_A: int, bits_phi: int, K_cap: int) -> List[int]:
    """Map each target bit to an increasing K (1..K_cap) whose code length is closest."""
    Ks = list(range(1, K_cap+1))
    # We don't know n_classes until we pick unique Ks; start by assuming number of targets
    n_classes_guess = max(1, len(bits_targets))
    # Compute code lengths per K given this guess
    K_to_bits = {K: _fourier_code_bits(K, F_size, bits_A, bits_phi, n_classes_guess) for K in Ks}
    chosen = []
    last_K = 0
    for bt in bits_targets:
        # pick K >= last_K with minimal |bits - bt|
        candidates = [(K, abs(K_to_bits[K] - bt)) for K in Ks if K > last_K]
        if not candidates:
            # if exhausted, just reuse last K_cap
            chosen.append(Ks[-1]); continue
        K_best = min(candidates, key=lambda kv: kv[1])[0]
        chosen.append(K_best)
        last_K = K_best
    # Recompute with true n_classes = len(set(chosen)) to make K_bits consistent
    n_classes = len(set(chosen))
    K_to_bits = {K: _fourier_code_bits(K, F_size, bits_A, bits_phi, n_classes) for K in set(chosen)}
    # Ensure strictly increasing bits by adjusting duplicates if needed
    uniq = []
    for K in chosen:
        if not uniq or K != uniq[-1]:
            uniq.append(K)
    return uniq

# ------------------------------
# AR & Piecewise (unchanged from v2)
# ------------------------------

def _stable_ar_coeffs(p: int, rng) -> np.ndarray:
    if p == 0:
        return np.array([], dtype=np.float32)
    angles = rng.uniform(0, np.pi, size=p)
    radii = rng.uniform(1.05, 2.0, size=p)
    roots = []
    for j in range(p//2):
        r = radii[j]; th = angles[j]
        roots.append(r*np.exp(1j*th)); roots.append(r*np.exp(-1j*th))
    if p % 2 == 1:
        roots.append(radii[-1])
    poly = np.poly(roots)
    a = -np.real(poly[1:]).astype(np.float32)
    return a

def gen_ar_suite(num_series: int, T: int, classes: List[int], seed: int = 0) -> List[SeriesSpec]:
    rng = np.random.default_rng(seed)
    out: List[SeriesSpec] = []
    bits_p = _bits_for_choice(len(classes))
    bits_coeff = _bits_for_real_grid(256)
    sid = 0
    per_class = max(1, num_series // len(classes))
    for p in classes:
        for _ in range(per_class):
            a = _stable_ar_coeffs(p, rng)
            K_bits = bits_p + p*bits_coeff
            L = T + 100
            e = rng.normal(0.0, 1.0, size=L).astype(np.float32)
            y = np.zeros(L, dtype=np.float32)
            for t in range(p, L):
                y[t] = np.dot(a, y[t-p:t][::-1]) + e[t]
            y = _zscore(y[-T:])
            out.append(SeriesSpec(y=y, K_bits=float(K_bits), family=f"AR{p}",
                                  params={"p": int(p)}, sid=sid)); sid += 1
    return out

def gen_piecewise_suite(num_series: int, T: int, classes: List[int], seed: int = 0) -> List[SeriesSpec]:
    rng = np.random.default_rng(seed)
    out: List[SeriesSpec] = []
    a_grid = np.linspace(-0.03, 0.03, 33)
    b_grid = np.linspace(-1.5, 1.5, 33)

    bits_S = _bits_for_choice(len(classes))
    bits_a = _bits_for_real_grid(len(a_grid))
    bits_b = _bits_for_real_grid(len(b_grid))

    sid = 0
    per_class = max(1, num_series // len(classes))
    for S in classes:
        for _ in range(per_class):
            min_gap = max(4, T // (3*max(S,1)))
            cps = []
            low, high = int(T*0.2), int(T*0.8)
            while len(cps) < max(S-1, 0):
                t0 = int(rng.integers(low, high))
                if all(abs(t0 - c) >= min_gap for c in cps):
                    cps.append(t0)
            cps = sorted(cps)
            a = [float(rng.choice(a_grid)) for _ in range(S)]
            b = [float(rng.choice(b_grid)) for _ in range(S)]
            t = np.arange(T, dtype=np.float32)
            y = np.zeros(T, dtype=np.float32)
            segs = [0] + cps + [T]
            for s in range(S):
                i0, i1 = segs[s], segs[s+1]
                ii = t[i0:i1]
                y[i0:i1] = a[s]*ii + b[s]
            y = _zscore(y)
            K_bits = bits_S + max(S-1,0)*_bits_for_int_range(int(T*0.2), int(T*0.8)-1) + S*(bits_a + bits_b)
            out.append(SeriesSpec(y=y, K_bits=float(K_bits), family=f"PW{S}",
                                  params={"S": int(S)}, sid=sid)); sid += 1
    return out

# Legacy mixed families if needed
def gen_legacy_suite(num_series: int, T: int, seed: int = 0) -> List[SeriesSpec]:
    rng = np.random.default_rng(seed)
    out: List[SeriesSpec] = []
    i = np.arange(T, dtype=np.float32)

    C_grid = np.linspace(-2.0, 2.0, 17)
    A_grid = np.linspace(0.2, 2.0, 10)
    a_grid = np.linspace(-0.02, 0.02, 17)
    b_grid = np.linspace(-1.0, 1.0, 17)
    f_grid = np.linspace(1/T, 0.25, 24)
    phi_grid = np.linspace(0, 2*np.pi, 16, endpoint=False)
    t0_grid = np.arange(int(T*0.2), int(T*0.8))
    STRUCT_BITS = _bits_for_choice(6)
    families = ["F1", "F2", "F3", "F4", "F5", "F6"]
    per_family = max(1, num_series // len(families))
    sid = 0

    # F1 constant
    for _ in range(per_family):
        c = rng.choice(C_grid)
        y = np.full(T, c, dtype=np.float32)
        K = STRUCT_BITS + _bits_for_real_grid(len(C_grid))
        out.append(SeriesSpec(_zscore(y), K, "F1", {"c": float(c)}, sid)); sid += 1
    # F2 linear
    for _ in range(per_family):
        a = rng.choice(a_grid); b = rng.choice(b_grid)
        y = a * i + b
        K = STRUCT_BITS + _bits_for_real_grid(len(a_grid)) + _bits_for_real_grid(len(b_grid))
        out.append(SeriesSpec(_zscore(y), K, "F2", {"a": float(a), "b": float(b)}, sid)); sid += 1
    # F3 sinusoid
    for _ in range(per_family):
        A = rng.choice(A_grid); f = rng.choice(f_grid); phi = rng.choice(phi_grid)
        y = A * np.sin(2*np.pi*f*i + phi)
        K = STRUCT_BITS + sum([_bits_for_real_grid(len(g)) for g in [A_grid, f_grid, phi_grid]])
        out.append(SeriesSpec(_zscore(y), K, "F3", {"A": float(A), "f": float(f), "phi": float(phi)}, sid)); sid += 1
    # F4 two sinusoids
    for _ in range(per_family):
        A1 = rng.choice(A_grid); f1 = rng.choice(f_grid); phi1 = rng.choice(phi_grid)
        A2 = rng.choice(A_grid); f2 = rng.choice(f_grid); phi2 = rng.choice(phi_grid)
        y = A1*np.sin(2*np.pi*f1*i + phi1) + A2*np.sin(2*np.pi*f2*i + phi2)
        K = STRUCT_BITS + 2*sum([_bits_for_real_grid(len(g)) for g in [A_grid, f_grid, phi_grid]])
        out.append(SeriesSpec(_zscore(y), K, "F4",
                              {"A1": float(A1), "f1": float(f1), "phi1": float(phi1),
                               "A2": float(A2), "f2": float(f2), "phi2": float(phi2)}, sid)); sid += 1
    # F5 piecewise linear
    for _ in range(per_family):
        a1 = rng.choice(a_grid); b1 = rng.choice(b_grid)
        a2 = rng.choice(a_grid); b2 = rng.choice(b_grid)
        t0 = int(rng.choice(t0_grid))
        y = np.where(i < t0, a1*i + b1, a2*i + b2)
        K = STRUCT_BITS + 2*(_bits_for_real_grid(len(a_grid)) + _bits_for_real_grid(len(b_grid))) + \
            _bits_for_int_range(int(T*0.2), int(T*0.8)-1)
        out.append(SeriesSpec(_zscore(y), K, "F5",
                              {"a1": float(a1), "b1": float(b1), "a2": float(a2), "b2": float(b2), "t0": int(t0)}, sid)); sid += 1
    # F6 modulated sinusoid
    for _ in range(per_family):
        a = rng.choice(a_grid); b = rng.choice(b_grid)
        A = rng.choice(A_grid); f = rng.choice(f_grid); phi = rng.choice(phi_grid)
        y = (a*i + b) * (A*np.sin(2*np.pi*f*i + phi))
        K = STRUCT_BITS + _bits_for_real_grid(len(a_grid)) + _bits_for_real_grid(len(b_grid)) + \
            _bits_for_real_grid(len(A_grid)) + _bits_for_real_grid(len(f_grid)) + _bits_for_real_grid(len(phi_grid))
        out.append(SeriesSpec(_zscore(y), K, "F6",
                              {"a": float(a), "b": float(b), "A": float(A), "f": float(f), "phi": float(phi)}, sid)); sid += 1

    while len(out) < num_series:
        base = out[rng.integers(0, len(out))]
        out.append(SeriesSpec(base.y.copy(), base.K_bits, base.family, dict(base.params), sid)); sid += 1

    return out

# ------------------------------
# Token-aware acceptance to enforce monotone difficulty (unchanged API)
# ------------------------------

def enforce_token_monotone(series: List[SeriesSpec], classes: List[int],
                           generator: str, context_len: int,
                           ctok_ref,  # ChronosTokenizerAdapter
                           metric: str = "runs",
                           oversample: float = 1.5) -> List[SeriesSpec]:
    key_pat = {
        "fourier": lambda fam: int(re.findall(r'FourierK(\d+)', fam)[0]),
        "ar": lambda fam: int(re.findall(r'AR(\d+)', fam)[0]),
        "piecewise": lambda fam: int(re.findall(r'PW(\d+)', fam)[0]),
        "legacy": lambda fam: {"F1":1,"F2":2,"F3":3,"F4":4,"F5":5,"F6":6}.get(fam, 0)
    }[generator]
    by_cls: Dict[int, List[SeriesSpec]] = {int(c): [] for c in classes}
    for s in series:
        try:
            k = key_pat(s.family)
            if k in by_cls:
                by_cls[k].append(s)
        except Exception:
            continue

    total_needed = sum(len(by_cls[int(c)]) for c in classes)
    nominal_per_class = int(total_needed / len(classes) / oversample)
    per_class_keep = max(1, nominal_per_class)

    metric_fn = runs_complexity if metric == "runs" else lz_complexity
    H = ctok_ref.pred_len
    class_metrics: Dict[int, List[Tuple[float, SeriesSpec]]] = {int(c): [] for c in classes}
    for c in classes:
        for s in by_cls[int(c)]:
            if len(s.y) < ctok_ref.pred_len + context_len:
                continue
            y_ctx = s.y[:context_len]
            y_lbl = s.y[context_len:context_len + H]
            ctx_ids, _, state = ctok_ref.tokenize_context(y_ctx)
            lbl_ids, _ = ctok_ref.tokenize_given_target(y_lbl, state)
            mval = metric_fn(lbl_ids)
            class_metrics[int(c)].append((mval, s))

    pooled = []
    for c in classes:
        pooled.extend([m for m, _ in class_metrics[int(c)]])
    if len(pooled) == 0:
        return [s for c in classes for s in by_cls[int(c)][:per_class_keep]]

    pooled = np.array(pooled, dtype=float)
    edges = np.quantile(pooled, np.linspace(0, 1, len(classes)+1))
    for j in range(1, len(edges)):
        if edges[j] <= edges[j-1]:
            edges[j] = edges[j-1] + 1e-6

    selected: List[SeriesSpec] = []
    for i, c in enumerate(classes):
        lo, hi = edges[i], edges[i+1]
        cand = [(m,s) for (m,s) in class_metrics[int(c)] if (m >= lo and m < hi)]
        expand = 0
        while len(cand) < per_class_keep and expand < 5:
            pad = (hi - lo) * 0.25 if (hi - lo) > 0 else 1.0
            cand = [(m,s) for (m,s) in class_metrics[int(c)] if (m >= lo - pad and m < hi + pad)]
            expand += 1
        cand.sort(key=lambda x: x[0])
        selected.extend([s for (_,s) in cand[:per_class_keep]])

    return selected

# ------------------------------
# Baseline scoring and summaries
# ------------------------------

def make_batches(arr: List, batch_size: int) -> List[List]:
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]

def bin_by_complexity(K_bits: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(K_bits, qs)
    for j in range(1, len(edges)):
        if edges[j] <= edges[j-1]:
            edges[j] = edges[j-1] + 1e-6
    bins = np.digitize(K_bits, edges[1:-1], right=False)
    return bins

def fit_slope_to_solomonoff(K_bits: np.ndarray, avg_logp: np.ndarray) -> Tuple[float, float]:
    X = np.vstack([np.ones_like(K_bits), K_bits]).T
    y = avg_logp
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, neg_beta = theta[0], theta[1]
    beta = -neg_beta
    return float(alpha), float(beta)

class ChronosEvaluator:
    def __init__(self, model_ids: Dict[str, str], device: str, batch_size: int):
        self.adapters: Dict[str, ChronosAdapter] = {name: ChronosAdapter(mid, device=device) for name, mid in model_ids.items()}
        self.batch_size = batch_size
    def baseline(self, series: List[SeriesSpec], context_len: int, temps: Optional[Dict[str, float]]) -> pd.DataFrame:
        rows = []
        # iterate models with a progress bar
        for name, adapter in tqdm(self.adapters.items(), desc="Models", total=len(self.adapters)):
            pred_len = adapter.ctok.pred_len

            # tokenize with a progress bar
            inputs, labels, masks, Ks, fams, sids, Hs = [], [], [], [], [], [], []
            for s in tqdm(series, desc=f"Tokenizing for {name}", leave=False):
                if len(s.y) < context_len + pred_len:
                    continue
                y_ctx = s.y[:context_len]
                y_tgt = s.y[context_len:context_len + pred_len]
                ctx_ids, ctx_mask, state = adapter.ctok.tokenize_context(y_ctx)
                lbl_ids, _ = adapter.ctok.tokenize_given_target(y_tgt, state)
                inputs.append(ctx_ids); labels.append(lbl_ids); masks.append(ctx_mask)
                Ks.append(s.K_bits); fams.append(s.family); sids.append(s.sid); Hs.append(pred_len)

            if not inputs:
                continue

            Tm = 1.0 if (temps is None or name not in temps) else float(temps[name])

            # batch scoring with a progress bar
            idx_list = list(range(len(inputs)))
            batches = [idx_list[i:i+self.batch_size] for i in range(0, len(idx_list), self.batch_size)]
            for idxs in tqdm(batches, desc=f"Scoring {name}", leave=False):
                x = torch.tensor([inputs[i] for i in idxs], dtype=torch.long, device=adapter.device)
                y = torch.tensor([labels[i] for i in idxs], dtype=torch.long, device=adapter.device)
                m = torch.tensor([masks[i]  for i in idxs], dtype=torch.long, device=adapter.device)
                token_nll, total_logprob = adapter.score_batch(x, y, m, temperature=Tm)
                for j, i in enumerate(idxs):
                    rows.append({
                        "model": name,
                        "sid": int(sids[i]),
                        "family": fams[i],
                        "K_bits": float(Ks[i]),
                        "C": int(context_len),
                        "H": int(Hs[i]),
                        "logprob": float(total_logprob[j].cpu().item()),
                        "nll_per_token": float(token_nll[j].cpu().item()),
                        "mode": "baseline_calibrated" if temps is not None else "baseline",
                    })
        return pd.DataFrame(rows)


def summarize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model_name, g in df.groupby("model"):
        K = g["K_bits"].values.astype(float)
        avg_logp = g["logprob"].values.astype(float) / g["H"].values.astype(float)
        alpha, beta = fit_slope_to_solomonoff(K, avg_logp)
        lo, hi = np.quantile(K, 0.2), np.quantile(K, 0.8)
        P_lo = np.mean(np.exp(avg_logp[K <= lo])); P_hi = np.mean(np.exp(avg_logp[K >= hi]))
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(K, -avg_logp)
        except Exception:
            rho = float("nan")
        out.append({"model": model_name, "alpha": alpha, "beta": beta,
                    "beta_target_ln2": math.log(2.0),
                    "SB_ratio_low_vs_high": P_lo/max(P_hi, 1e-12),
                    "Spearman_K_vs_minusNLL": rho,
                    "n_samples": len(g)})
    return pd.DataFrame(out)

def plot_baseline(df: pd.DataFrame, out_path: str, title: str, plot_bins: Optional[int] = None):
    '''
    Plots average log-probability per token vs. complexity (bits) with 95% CI error bars.
    CI is normal-approximation: 1.96 * sd / sqrt(n) within each complexity bin.
    '''
    plt.figure(figsize=(7,5))
    for m, g in df.groupby("model"):
        unique_vals = len(np.unique(g["K_bits"].values))
        n_bins = plot_bins if (plot_bins is not None and plot_bins > 0) else min(max(unique_vals, 6), 24)

        bins = bin_by_complexity(g["K_bits"].values, n_bins=n_bins)
        xs, means, yerr_low, yerr_high = [], [], [], []

        for b in sorted(np.unique(bins)):
            mask = (bins == b)
            Kb = g["K_bits"].values[mask]
            Lb = g["logprob"].values[mask] / g["H"].values[mask]  # avg logp per token
            if Lb.size == 0:
                continue
            mu = float(np.mean(Lb))
            if Lb.size >= 2:
                sd = float(np.std(Lb, ddof=1))
                se = sd / max(1.0, np.sqrt(Lb.size))
                ci = 1.96 * se
            else:
                ci = 0.0

            xs.append(float(np.mean(Kb)))
            means.append(mu)
            yerr_low.append(ci)
            yerr_high.append(ci)

        xs = np.array(xs, dtype=float)
        means = np.array(means, dtype=float)
        yerr = np.vstack([np.array(yerr_low, dtype=float), np.array(yerr_high, dtype=float)])

        plt.errorbar(xs, means, yerr=yerr, fmt='-o', capsize=3, label=m)

    plt.xlabel("Complexity K(s) (bits)")
    plt.ylabel("Avg log-prob per token")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def calibrate_temperatures(model_ids: Dict[str, str], neutral_series: List[SeriesSpec],
                           context_len: int, batch_size: int, device: str,
                           target: Optional[float] = None):
    adapters: Dict[str, ChronosAdapter] = {name: ChronosAdapter(mid, device=device) for name, mid in model_ids.items()}
    base_nll = {}
    for name, adapter in adapters.items():
        pred_len = adapter.ctok.pred_len
        inputs, labels, masks = [], [], []
        for s in neutral_series:
            if len(s.y) < context_len + pred_len:
                continue
            y_ctx = s.y[:context_len]; y_tgt = s.y[context_len:context_len+pred_len]
            ctx_ids, ctx_mask, state = adapter.ctok.tokenize_context(y_ctx)
            lbl_ids, _ = adapter.ctok.tokenize_given_target(y_tgt, state)
            inputs.append(ctx_ids); labels.append(lbl_ids); masks.append(ctx_mask)
        if not inputs:
            base_nll[name] = float("nan"); continue
        nlls = []
        for idxs in make_batches(list(range(len(inputs))), batch_size):
            x = torch.tensor([inputs[i] for i in idxs], dtype=torch.long, device=adapter.device)
            y = torch.tensor([labels[i] for i in idxs], dtype=torch.long, device=adapter.device)
            m = torch.tensor([masks[i]  for i in idxs], dtype=torch.long, device=adapter.device)
            token_nll, _ = adapter.score_batch(x, y, m, temperature=1.0)
            nlls.extend(token_nll.cpu().numpy().tolist())
        base_nll[name] = float(np.mean(nlls))

    if target is None:
        vals = [v for v in base_nll.values() if np.isfinite(v)]
        target = float(np.mean(vals)) if vals else 1.0

    temps = {}
    for name, adapter in adapters.items():
        if not np.isfinite(base_nll[name]):
            temps[name] = 1.0; continue
        pred_len = adapter.ctok.pred_len
        inputs, labels, masks = [], [], []
        for s in neutral_series:
            if len(s.y) < context_len + pred_len:
                continue
            y_ctx = s.y[:context_len]; y_tgt = s.y[context_len:context_len+pred_len]
            ctx_ids, ctx_mask, state = adapter.ctok.tokenize_context(y_ctx)
            lbl_ids, _ = adapter.ctok.tokenize_given_target(y_tgt, state)
            inputs.append(ctx_ids); labels.append(lbl_ids); masks.append(ctx_mask)
        if not inputs:
            temps[name] = 1.0; continue

        def mean_nll_at(T):
            nlls = []
            for idxs in make_batches(list(range(len(inputs))), batch_size):
                x = torch.tensor([inputs[i] for i in idxs], dtype=torch.long, device=adapter.device)
                y = torch.tensor([labels[i] for i in idxs], dtype=torch.long, device=adapter.device)
                m = torch.tensor([masks[i]  for i in idxs], dtype=torch.long, device=adapter.device)
                token_nll, _ = adapter.score_batch(x, y, m, temperature=T)
                nlls.extend(token_nll.cpu().numpy().tolist())
            return float(np.mean(nlls))

        lo, hi = 0.3, 3.0
        for _ in range(20):
            mid = 0.5*(lo+hi)
            val = mean_nll_at(mid)
            if val < target:  # too low NLL -> increase T
                lo = mid
            else:             # too high NLL -> decrease T
                hi = mid
        temps[name] = 0.5*(lo+hi)

    return temps, base_nll, target

def iso_difficulty_slice(baseline_df: pd.DataFrame, low_p: float = 0.4, high_p: float = 0.6) -> pd.DataFrame:
    tab = baseline_df.pivot_table(index="sid", columns="model", values="nll_per_token", aggfunc="mean")
    keep_mask = pd.Series(True, index=tab.index)
    for m in tab.columns:
        col = tab[m].dropna()
        if col.empty:
            continue
        lo = col.quantile(low_p); hi = col.quantile(high_p)
        keep_mask &= tab[m].between(lo, hi)
    sids_keep = set(tab.index[keep_mask])
    return baseline_df[baseline_df["sid"].isin(sids_keep)].copy()

def summarize_iso(df_iso: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, g in df_iso.groupby("model"):
        K = g["K_bits"].values.astype(float)
        avg_logp = g["logprob"].values.astype(float) / g["H"].values.astype(float)
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(K, -avg_logp)
        except Exception:
            rho = float("nan")
        lo, hi = np.quantile(K, 0.2), np.quantile(K, 0.8)
        P_lo = np.mean(np.exp(avg_logp[K <= lo])); P_hi = np.mean(np.exp(avg_logp[K >= hi]))
        out.append({"model": m, "iso_Spearman_K_vs_minusNLL": rho,
                    "iso_SB_ratio_low_vs_high": P_lo/max(P_hi, 1e-12),
                    "n_samples_iso": len(g)})
    return pd.DataFrame(out)

def evaluate_occam_pairs(model_ids: Dict[str, str], pairs: List[OccamPair],
                         batch_size: int, device: str,
                         temps: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adapters: Dict[str, ChronosAdapter] = {name: ChronosAdapter(mid, device=device) for name, mid in model_ids.items()}
    rows = []
    for name, adapter in adapters.items():
        pred_len = adapter.ctok.pred_len
        ctx_ids_list, ctx_masks_list = [], []
        lblS_list, lblC_list = [], []
        meta = []
        for P in pairs:
            if len(P.y_simple_future) < pred_len or len(P.y_complex_future) < pred_len:
                continue
            ctx_ids, ctx_mask, state = adapter.ctok.tokenize_context(P.y_context)
            lblS_ids, _ = adapter.ctok.tokenize_given_target(P.y_simple_future[:pred_len], state)
            lblC_ids, _ = adapter.ctok.tokenize_given_target(P.y_complex_future[:pred_len], state)

            ctx_ids_list.append(ctx_ids); ctx_masks_list.append(ctx_mask)
            lblS_list.append(lblS_ids); lblC_list.append(lblC_ids)
            meta.append((P.sid, P.family, P.K_simple, P.K_complex, P.delta_K))

        if not ctx_ids_list:
            continue

        Tm = 1.0 if temps is None or name not in temps else float(temps[name])

        for idxs in make_batches(list(range(len(ctx_ids_list))), batch_size):
            x = torch.tensor([ctx_ids_list[i] for i in idxs], dtype=torch.long, device=adapter.device)
            m = torch.tensor([ctx_masks_list[i] for i in idxs], dtype=torch.long, device=adapter.device)
            yS = torch.tensor([lblS_list[i] for i in idxs], dtype=torch.long, device=adapter.device)
            yC = torch.tensor([lblC_list[i] for i in idxs], dtype=torch.long, device=adapter.device)

            _, logpS = adapter.score_batch(x, yS, m, temperature=Tm)
            _, logpC = adapter.score_batch(x, yC, m, temperature=Tm)

            for j, i in enumerate(idxs):
                sid, fam, Ks, Kc, dK = meta[i]
                rows.append({
                    "model": name, "sid": int(sid), "family": fam,
                    "K_simple": float(Ks), "K_complex": float(Kc), "delta_K": float(dK),
                    "logprob_simple": float(logpS[j].cpu().item()),
                    "logprob_complex": float(logpC[j].cpu().item()),
                    "delta_logprob": float((logpS[j] - logpC[j]).cpu().item()),
                    "H": int(pred_len)
                })

    df = pd.DataFrame(rows)
    out = []
    for m, g in df.groupby("model"):
        win_rate = float(np.mean((g["delta_logprob"].values > 0).astype(np.float32))) if len(g) else float("nan")
        if len(g) >= 2:
            X = np.vstack([np.ones_like(g["delta_K"].values), g["delta_K"].values]).T
            y = g["delta_logprob"].values
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, slope = theta[0], theta[1]
        else:
            alpha, slope = float("nan"), float("nan")
        out.append({"model": m, "occam_win_rate": win_rate,
                    "occam_mean_delta_logprob": float(np.mean(g["delta_logprob"].values)),
                    "occam_slope_vs_deltaK": float(slope), "n_pairs": len(g)})
    summ = pd.DataFrame(out)
    return df, summ

def plot_occam_winrate(df: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(7,5))
    for m, g in df.groupby("model"):
        qs = np.linspace(0, 1, 9)
        edges = np.quantile(g["delta_K"].values, qs)
        for j in range(1, len(edges)):
            if edges[j] <= edges[j-1]:
                edges[j] = edges[j-1] + 1e-6
        bins = np.digitize(g["delta_K"].values, edges[1:-1], right=False)
        xs, ys = [], []
        for b in sorted(np.unique(bins)):
            dKb = g["delta_K"].values[bins==b]
            wr  = np.mean((g["delta_logprob"].values[bins==b] > 0).astype(np.float32))
            xs.append(np.mean(dKb)); ys.append(wr)
        xs = np.array(xs); ys = np.array(ys)
        plt.plot(xs, ys, marker='o', label=m)
    plt.xlabel("ΔK (complex - simple) (bits)"); plt.ylabel("Occam win rate")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_chronos_sb")
    ap.add_argument("--models", type=str, nargs="*", default=["tiny","mini","small","base","large"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)

    # Baseline sweep
    ap.add_argument("--generator", type=str, default="fourier", choices=["fourier","ar","piecewise","legacy"])
    ap.add_argument("--classes", type=str, default="1,2,3,4,5,6",
                    help="For fourier: K list if no bits grid. For ar: p list. For piecewise: S list.")
    ap.add_argument("--fourier_bits_grid", type=str, default="", help="E.g., '10:100:10' or '10,20,30,...'")
    ap.add_argument("--fourier_k_cap", type=int, default=16, help="Max K to consider when mapping target bits.")
    ap.add_argument("--F_size", type=int, default=64, help="Fourier frequency pool size.")
    ap.add_argument("--num_series", type=int, default=2400)
    ap.add_argument("--series_len", type=int, default=256)
    ap.add_argument("--context_len", type=int, default=128)

    # Token-aware monotonic enforcement
    ap.add_argument("--enforce_token_monotone", action="store_true")
    ap.add_argument("--token_metric", type=str, default="runs", choices=["runs","lz"])
    ap.add_argument("--oversample", type=float, default=1.5)
    ap.add_argument("--accept_ref_model", type=str, default="large")

    # Plotting
    ap.add_argument("--plot_bins", type=int, default=0, help="Override bin count for baseline plot (0=auto).")

    # Occam pairs
    ap.add_argument("--occam_pairs", action="store_true")
    ap.add_argument("--num_pairs", type=int, default=1500)

    # Calibration
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib_num", type=int, default=400)

    # Iso-difficulty
    ap.add_argument("--iso_difficulty", action="store_true")
    ap.add_argument("--iso_low", type=float, default=0.4)
    ap.add_argument("--iso_high", type=float, default=0.6)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chosen = {k: MODEL_ID_MAP[k] for k in args.models if k in MODEL_ID_MAP}
    if not chosen:
        raise ValueError("No valid model keys. Choose among: " + ", ".join(MODEL_ID_MAP.keys()))

    # Determine H_max for Occam pairs
    pred_lens = []
    for name, mid in chosen.items():
        cfg = AutoConfig.from_pretrained(mid)
        ctok_tmp = ChronosTokenizerAdapter(cfg)
        pred_lens.append(ctok_tmp.pred_len)
    H_max = int(max(pred_lens)) if pred_lens else 64

    # Reference tokenizer for acceptance
    if args.enforce_token_monotone:
        ref_id = MODEL_ID_MAP.get(args.accept_ref_model, args.accept_ref_model)
        ref_cfg = AutoConfig.from_pretrained(ref_id)
        ctok_ref = ChronosTokenizerAdapter(ref_cfg)
    else:
        ctok_ref = None  # type: ignore

    # Build series per generator
    if args.generator == "fourier":
        # decide classes (K list) either from bits grid or explicit K list
        A_grid = np.linspace(0.3, 1.8, 16)
        phi_grid = np.linspace(0, 2*np.pi, 32, endpoint=False)
        bits_A = _bits_for_real_grid(len(A_grid))
        bits_phi = _bits_for_real_grid(len(phi_grid))

        bits_targets = parse_bits_grid(args.fourier_bits_grid)
        if bits_targets:
            K_list = pick_K_for_bits_targets(bits_targets, args.series_len, args.F_size, bits_A, bits_phi, args.fourier_k_cap)
        else:
            K_list = [int(x) for x in args.classes.split(',') if x.strip()]
        base_series = gen_fourier_suite(int(args.num_series*args.oversample), args.series_len, K_list, seed=args.seed, F_size=args.F_size)
        classes_for_enforce = K_list[:]  # K values used as class keys
    elif args.generator == "ar":
        classes = [int(x) for x in args.classes.split(',') if x.strip()]
        base_series = gen_ar_suite(int(args.num_series*args.oversample), args.series_len, classes, seed=args.seed)
        classes_for_enforce = classes[:]
    elif args.generator == "piecewise":
        classes = [int(x) for x in args.classes.split(',') if x.strip()]
        base_series = gen_piecewise_suite(int(args.num_series*args.oversample), args.series_len, classes, seed=args.seed)
        classes_for_enforce = classes[:]
    else:
        base_series = gen_legacy_suite(int(args.num_series*args.oversample), args.series_len, seed=args.seed)
        classes_for_enforce = [1,2,3,4,5,6]

    # Token-aware acceptance
    if args.enforce_token_monotone and ctok_ref is not None:
        series = enforce_token_monotone(base_series, classes_for_enforce, args.generator, args.context_len,
                                        ctok_ref, metric=args.token_metric, oversample=args.oversample)
        if len(series) < args.num_series // 2:
            series = base_series[:args.num_series]
    else:
        series = base_series[:args.num_series]

    # Evaluate
    evaluator = ChronosEvaluator(chosen, device=args.device, batch_size=args.batch_size)

    temps = None
    base_nll_unscaled = None
    calib_target = None

    if args.calibrate:
        # Build neutral set (use simple K spread)
        if args.generator == "fourier":
            neutral_Ks = sorted(set([1,2,3,4,5,6,8,10]))
            neutral_series = gen_fourier_suite(args.calib_num, args.context_len + H_max, neutral_Ks, seed=args.seed+7, F_size=args.F_size)
        elif args.generator == "ar":
            neutral_series = gen_ar_suite(args.calib_num, args.context_len + H_max, [0,1,2,3,4,5], seed=args.seed+7)
        elif args.generator == "piecewise":
            neutral_series = gen_piecewise_suite(args.calib_num, args.context_len + H_max, [1,2,3,4,5,6], seed=args.seed+7)
        else:
            neutral_series = gen_legacy_suite(args.calib_num, args.context_len + H_max, seed=args.seed+7)

        temps, base_nll_unscaled, calib_target = calibrate_temperatures(
            chosen, neutral_series, context_len=args.context_len,
            batch_size=args.batch_size, device=args.device
        )
        with open(os.path.join(args.out_dir, "calibration_temps.json"), "w") as f:
            json.dump({"temps": temps, "base_nll_unscaled": base_nll_unscaled, "target_nll": calib_target}, f, indent=2)

    df_base = evaluator.baseline(series, context_len=args.context_len, temps=temps)
    df_base.to_csv(os.path.join(args.out_dir, "baseline_scores.csv"), index=False)
    summ_base = summarize_baseline(df_base)
    summ_base.to_csv(os.path.join(args.out_dir, "baseline_summary.csv"), index=False)
    plot_baseline(df_base, os.path.join(args.out_dir, "baseline_plot.png"),
                  f"Avg log-prob/token vs. K (generator={args.generator}{' + calibrated' if args.calibrate else ''})",
                  plot_bins=args.plot_bins)

    if args.iso_difficulty:
        df_iso = iso_difficulty_slice(df_base, low_p=args.iso_low, high_p=args.iso_high)
        df_iso.to_csv(os.path.join(args.out_dir, "iso_slice.csv"), index=False)
        summ_iso = summarize_iso(df_iso)
        summ_iso.to_csv(os.path.join(args.out_dir, "iso_summary.csv"), index=False)

    if args.occam_pairs:
        pairs = _make_occam_pairs(args.num_pairs, C=args.context_len, H_max=H_max, T_total=args.series_len, seed=args.seed+13)
        df_occam, summ_occam = evaluate_occam_pairs(chosen, pairs, batch_size=args.batch_size,
                                                    device=args.device, temps=temps)
        df_occam.to_csv(os.path.join(args.out_dir, "occam_pairs_scores.csv"), index=False)
        summ_occam.to_csv(os.path.join(args.out_dir, "occam_pairs_summary.csv"), index=False)
        plot_occam_winrate(df_occam, os.path.join(args.out_dir, "occam_winrate.png"),
                           "Occam win rate vs. ΔK")

    meta = {"MODEL_ID_MAP": chosen, "args": vars(args),
            "note": "v3 with Fourier target bits and adaptive plot bins"}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()
