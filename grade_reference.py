"""
GRADE on cognitive-distortion validation in OLMo.

Implements the GRADE method (Wang et al. 2026, arXiv:2604.02830) on the
clinical-sycophancy stimuli to answer:

  Does the model have the CAPACITY to produce therapeutic (CBT-reframing)
  continuations to distorted user inputs, or does it merely fail to
  default to them?

GRADE quantifies the knowledge gap for a given query via the
cross-layer rank ratio between gradients and hidden states:

                  srank(C_g^(l))
    RankRatio_l = ---------------
                  srank(C_h^(l))

where, at MLP layer l:

    h ∈ R^{n × d_ff}     MLP intermediate states (SwiGLU output, input of
                         down_proj), one row per completion token.
    g ∈ R^{d_model × d_ff}   gradient of the completion log-loss
                             L_pos = -Σ_t log p(y_t | y_<t, q)
                             w.r.t. the down_proj weight (Eq. 4).
    C_h = h h^T ∈ R^{n × n}                (Gram matrix of h)
    C_g = C_h^+ (h g^T g h^T) C_h^+ ∈ R^{n × n}    (projected gradient
                                                   covariance, Eq. 3.2.1)

Stable rank (Eq. 6) is used to avoid hard-threshold instability:

    srank_pre(M) = Σ λ_i(M) / λ_1(M)
    srank_pos(M) = Σ (λ_i(M))^2 / (λ_1(M))^2

with λ_i the singular values of M (for symmetric PSD C_h, C_g these are
the eigenvalues). Both variants are reported.

We use an efficient identity to avoid ever materialising C_g as an
n×n matrix. Because C_g = (C_h^+ h g^T)(g h^T C_h^+) = M M^T with
M = C_h^+ h g^T ∈ R^{n × d_model}, the eigenvalues of C_g are the
squared singular values of M. We SVD M directly (small: n ≤ ~300,
d_model = 2048).

Experiments:

  G1  Per-layer rank ratio for therapeutic vs sycophantic completions
      on distortion stimuli, paired by stimulus. Tests whether
      therapeutic targets require more knowledge updates than
      sycophantic targets under the same query.

  G5  Capacity probe. Defines a per-stimulus, per-target capacity score

            C(x, target) = 1 / RankRatio_pos(x, target)   (mean over layers)

      and compares:
        * C_ther(x) vs C_syc(x) on distortion stimuli
        * C_ther(x)  on distortion stimuli vs the same completion key
                    on factual-control stimuli (pure-knowledge baseline)
      A model with HIGH therapeutic capacity has C_ther ≈ C_syc ≈ C_factual.
      A CAPACITY GAP for therapeutic reframing manifests as
      C_ther(distortion) < C_syc(distortion) AND
      C_ther(distortion) < C_ther(factual_mock).

  G3  Mechanism-only steering from gradient subspace.
      For each layer we form the per-stimulus residual-stream gradient
      contrast Δg(x) = ∇_{h_res} L_T - ∇_{h_res} L_S, stack into matrix
      (N, d_model), and use its top right-singular vector as a steering
      direction v*. We measure the shift in teacher-forced
      log P(therapeutic) - log P(sycophantic) under negative-steering
      with v* and compare to the baseline activation-based direction d
      from the existing repo (reference.py). Predicts: v* attains a
      higher specificity ratio |ΔE5|/|ΔE6| than d because d mixes style
      into its top component.

  G4  Consensus subspace sharpening across the 12 CBT subtypes.
      For each subtype we compute its per-layer RankRatio profile.
      Stacking the 12 profiles gives a (12, L) matrix; its stable rank
      quantifies how shared the therapeutic-capacity gap is across
      subtypes, independent of the repo's activation-subspace geometry.

Usage:
    python grade_reference.py --quick         # smoke test (5–8 min CPU/MPS)
    python grade_reference.py                 # default 1B local run
    python grade_reference.py --model 7b      # 7B (for Colab)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse repo utilities (activation extraction, directions, hooks,
# teacher-forced logprob, intervention hooks).
from reference import (
    SEED, STIM_DIR, MODEL_IDS,
    set_seeds, select_device, get_device, cleanup,
    load_json, save_json, format_prompt,
    extract_paired, contrastive_direction,
    completion_logprob, compute_baseline_signals,
    compute_intervention_signals, shift_summary,
    negative_steering_hook, pick_layers, stratified_sample,
)

# DPO variant of OLMo-3 7B for Colab runs (user-requested).
MODEL_IDS = {**MODEL_IDS, "7b-dpo": "allenai/Olmo-3-7B-Instruct-DPO"}

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
REVIEW_DIR = ROOT / "review"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
REVIEW_DIR.mkdir(exist_ok=True)

COLOR = {
    "ther": "#2980b9",
    "syc": "#c0392b",
    "fact": "#27ae60",
    "cap": "#8e44ad",
    "neutral": "#7f8c8d",
}

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Stable rank and rank-ratio primitives (GRADE §3.2)


def stable_ranks(eigvals: torch.Tensor) -> Dict[str, float]:
    """Eq. 6: srank_pre = Σ λ_i / λ_1; srank_pos = Σ (λ_i / λ_1)^2.

    eigvals: non-negative singular values of a covariance-shaped matrix,
    or equivalently σ_i^2 for a rectangular matrix. Sorted or unsorted.
    """
    eig = eigvals.to(torch.float64).clamp(min=0.0)
    if eig.numel() == 0:
        return {"pre": 0.0, "pos": 0.0}
    lam1 = eig.max().clamp(min=1e-20)
    return {
        "pre": float((eig / lam1).sum()),
        "pos": float((eig / lam1).pow(2).sum()),
    }


def rank_ratio_from_h_g(h: torch.Tensor, g_w: torch.Tensor,
                        tol_rel: float = 1e-12) -> Dict[str, float]:
    """Projected rank ratios for one (layer, stimulus).

    h:    (n, d_ff)      float64. MLP intermediate for completion tokens.
    g_w:  (d_model, d_ff) float64. Gradient of completion loss w.r.t. W_down.

    Returns a dict with srank_{pre,pos}_h, srank_{pre,pos}_g, and
    rank_ratio_{pre,pos}. Uses the identity
            C_g = M M^T  with  M = C_h^+ h g_w^T    (n × d_model)
    so we SVD M directly (cheap: n is completion token count).
    """
    h = h.to(torch.float64)
    g_w = g_w.to(torch.float64)
    n = h.shape[0]

    # C_h = h h^T is symmetric PSD.
    C_h = h @ h.T  # (n, n)
    eig_h, evec_h = torch.linalg.eigh(C_h)  # ascending
    eig_h = eig_h.clamp(min=0.0)

    tol = max(eig_h.max().item() * tol_rel, 1e-20)
    inv_eig = torch.where(eig_h > tol, 1.0 / eig_h.clamp(min=tol), torch.zeros_like(eig_h))
    # C_h^+ = V diag(1/λ_i, trunc) V^T
    C_h_pinv = (evec_h * inv_eig.unsqueeze(0)) @ evec_h.T  # (n, n)

    # M = C_h^+ h g_w^T ∈ R^{n × d_model}
    M = C_h_pinv @ (h @ g_w.T)

    sigma = torch.linalg.svdvals(M)  # (min(n, d_model),)
    # eigenvalues of C_g = M M^T are σ_i(M)^2
    eig_g = sigma.pow(2)

    sr_h = stable_ranks(eig_h)
    sr_g = stable_ranks(eig_g)
    return {
        "srank_pre_h": sr_h["pre"],
        "srank_pos_h": sr_h["pos"],
        "srank_pre_g": sr_g["pre"],
        "srank_pos_g": sr_g["pos"],
        "rank_ratio_pre": sr_g["pre"] / max(sr_h["pre"], 1e-12),
        "rank_ratio_pos": sr_g["pos"] / max(sr_h["pos"], 1e-12),
    }


# ---------------------------------------------------------------------------
# Gradient + MLP-intermediate capture


def _layer_params(model, layers: List[int]):
    return [model.model.layers[l].mlp.down_proj.weight for l in layers]


def extract_mlp_grad_data(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    layers: List[int],
    residual_layers: List[int] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor],
           Dict[int, torch.Tensor], Dict[int, torch.Tensor], float, int]:
    """Run one forward + backward on (prompt, completion). Capture:

      h[l]        (n, d_ff)          MLP intermediate (input of down_proj)
                                      restricted to completion token positions
      g[l]        (d_model, d_ff)    ∇_{W_down^(l)} L_pos
      residual_grad[l]  (n, d_model)  ∇_{h^(l)_res} L_pos, for residual-stream
                                      steering use; only for `residual_layers`
      residual_h[l]  (n, d_model)    residual stream at completion positions
                                      (for span and steering analysis)
      loss (float), n_completion_tokens (int)
    """
    residual_layers = residual_layers or []
    device = get_device(model)

    formatted = format_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(formatted, return_tensors="pt")
    full_ids = tokenizer.encode(formatted + completion, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]
    assert prompt_len <= full_ids.shape[1] and (
        full_ids[0, :prompt_len].tolist() == prompt_ids[0].tolist()
    ), "tokenizer retokenising across boundary"
    n_comp = int(full_ids.shape[1] - prompt_len)
    full_ids = full_ids.to(device)

    # Capture MLP intermediate = input of down_proj at each target layer.
    captured_h = {}
    # Capture residual-stream tensor (output of decoder block) for requested layers.
    captured_res = {}
    handles = []

    def mlp_pre_hook(idx):
        def fn(module, inputs):
            captured_h[idx] = inputs[0]  # (1, seq, d_ff)
        return fn

    def res_hook(idx):
        def fn(module, inp, out):
            h_out = out[0] if isinstance(out, tuple) else out
            h_out.retain_grad()
            captured_res[idx] = h_out
        return fn

    for l in layers:
        handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(mlp_pre_hook(l)))
    for l in residual_layers:
        handles.append(model.model.layers[l].register_forward_hook(res_hook(l)))

    try:
        if n_comp == 0:
            raise ValueError(f"empty completion for prompt: {prompt[:60]!r}")
        model.zero_grad(set_to_none=True)
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq, V)
        pred_logits = logits[0, prompt_len - 1 : prompt_len - 1 + n_comp].float()
        target_ids = full_ids[0, prompt_len : prompt_len + n_comp]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_lps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        loss = -token_lps.sum()  # L_pos per paper Eq. 3 (sum over completion tokens)

        grad_targets = _layer_params(model, layers)
        res_targets = [captured_res[l] for l in residual_layers]
        grads = torch.autograd.grad(
            loss, grad_targets + res_targets, retain_graph=False, create_graph=False,
        )
    finally:
        for h in handles:
            h.remove()

    mlp_h: Dict[int, torch.Tensor] = {}
    mlp_g: Dict[int, torch.Tensor] = {}
    for i, l in enumerate(layers):
        h_l = captured_h[l][0, prompt_len : prompt_len + n_comp].detach().cpu().float()
        g_l = grads[i].detach().cpu().float()
        mlp_h[l] = h_l
        mlp_g[l] = g_l

    res_grad_out: Dict[int, torch.Tensor] = {}
    res_h_out: Dict[int, torch.Tensor] = {}
    for j, l in enumerate(residual_layers):
        g_r = grads[len(layers) + j][0, prompt_len : prompt_len + n_comp].detach().cpu().float()
        h_r = captured_res[l][0, prompt_len : prompt_len + n_comp].detach().cpu().float()
        res_grad_out[l] = g_r
        res_h_out[l] = h_r

    return mlp_h, mlp_g, res_h_out, res_grad_out, float(loss.detach().item()), n_comp


def extract_paired_grad_data(
    model, tokenizer,
    stimuli: List[dict],
    key_T: str, key_S: str,
    mlp_layers: List[int],
    residual_layers: List[int],
    desc: str = "grad-extract",
):
    """One fwd+bwd per (stimulus, role). Compute rank ratios inline and
    discard the heavy h/g tensors immediately to keep memory bounded.

    Returns a list of dicts (one per stimulus) with per-role:
        'ranks': {layer: {srank_pre_h, srank_pos_h, srank_pre_g, srank_pos_g,
                          rank_ratio_pre, rank_ratio_pos}}
        'res_g': {layer: (n_tok, d_model) float32}   # residual_layers only
        'loss': float
        'n_tok': int
    """
    out = []
    for i, s in enumerate(tqdm(stimuli, desc=desc)):
        per = {}
        for role, key in (("T", key_T), ("S", key_S)):
            mlp_h, mlp_g, _, res_g, loss, n_tok = extract_mlp_grad_data(
                model, tokenizer, s["user_prompt"], s[key], mlp_layers, residual_layers,
            )
            ranks = {}
            for l in mlp_layers:
                ranks[l] = rank_ratio_from_h_g(mlp_h[l], mlp_g[l])
            # free heavy tensors immediately
            del mlp_h, mlp_g
            per[role] = {"ranks": ranks, "res_g": res_g,
                         "loss": loss, "n_tok": n_tok}
        out.append(per)
        cleanup()
    return out


# ---------------------------------------------------------------------------
# G1 + G5: rank ratios per layer per stimulus


def per_stim_rank_ratios(
    records: List[dict], layers: List[int], role: str = "T",
) -> Dict[int, List[Dict[str, float]]]:
    """Pull precomputed ranks out of the records."""
    out = {l: [] for l in layers}
    for rec in records:
        for l in layers:
            out[l].append(rec[role]["ranks"][l])
    return out


def _t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided survival function of Student-t with df degrees of freedom.

    Uses scipy.stats.t when available (sklearn is already a dep of the repo,
    so scipy is present). Falls back to a regularised-incomplete-beta
    implementation if scipy import fails.
    """
    if df < 1:
        return float("nan")
    try:
        from scipy.stats import t as _t  # type: ignore
        return float(2.0 * _t.sf(abs(t), df))
    except Exception:
        # Fallback via regularised incomplete beta: P(|T| > t)
        # = I_{df/(df+t^2)}(df/2, 1/2). Implemented via math.lgamma.
        from math import lgamma, log, exp
        x = df / (df + t * t)
        a, b = df / 2.0, 0.5
        # Use continued-fraction for I_x(a, b); for our n<=36 this converges fast.
        # Here we just use the identity via symmetry as a last resort.
        def betacf(a, b, x, itmax=200, eps=1e-12):
            qab = a + b; qap = a + 1.0; qam = a - 1.0
            c = 1.0; d = 1.0 - qab * x / qap
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            h = d
            for m in range(1, itmax + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                h = h * d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h
        # I_x(a, b) = x^a (1-x)^b / (a B(a,b)) * betacf(a,b,x)
        if x < (a + 1.0) / (a + b + 2.0):
            bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
            I = bt * betacf(a, b, x) / a
        else:
            bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
            I = 1.0 - bt * betacf(b, a, 1.0 - x) / b
        return float(I)


def paired_tests(a: List[float], b: List[float]) -> Dict[str, float]:
    """Paired t-test (Student-t df=n-1), sign test, and Cohen's d_z.

    Returns NaN for t and both p-values when n < 2 or se == 0 (guards the
    n=1 degenerate path flagged by review).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = int(len(d))
    mean_d = float(d.mean()) if n else 0.0
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    if n < 2 or not np.isfinite(se) or se == 0.0:
        return {
            "n": n, "mean_diff": mean_d, "sd": sd, "se": se,
            "t": float("nan"), "p_t": float("nan"),
            "cohens_dz": float("nan"),
            "p_sign": float("nan"),
            "n_positive_diff": int((d > 0).sum()) if n else 0,
        }
    t = mean_d / se
    p_t = _t_sf_two_sided(t, df=n - 1)
    dz = mean_d / sd  # Cohen's d_z for paired samples
    # Two-sided exact binomial sign test, null = 0.5
    from math import comb
    n_pos = int((d > 0).sum())
    n_nz = int((d != 0).sum())
    if n_nz == 0:
        p_sign = 1.0
    else:
        # Two-sided p-value: P(|X - n/2| >= |n_pos - n/2|) under X ~ Bin(n_nz, 0.5)
        k = n_pos
        target = abs(k - n_nz / 2.0)
        total = 0
        for i in range(n_nz + 1):
            if abs(i - n_nz / 2.0) >= target - 1e-12:
                total += comb(n_nz, i)
        p_sign = float(total / (2 ** n_nz))
    return {"n": n, "mean_diff": mean_d, "sd": sd, "se": se,
            "t": float(t), "p_t": float(p_t),
            "cohens_dz": float(dz),
            "p_sign": float(p_sign),
            "n_positive_diff": n_pos}


def bootstrap_ci_mean_diff(a: List[float], b: List[float],
                            n_boot: int = 2000, seed: int = SEED) -> Dict[str, float]:
    a = np.asarray(a); b = np.asarray(b)
    d = a - b
    rng = np.random.RandomState(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boots[i] = d[idx].mean()
    return {
        "mean_diff": float(d.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_boot": int(n_boot),
    }


def cluster_bootstrap_ci_mean_diff(
    a: List[float], b: List[float], cluster_ids: List,
    n_boot: int = 2000, seed: int = SEED,
) -> Dict[str, float]:
    """Paired cluster bootstrap: resample CLUSTERS (subtypes) with
    replacement, then include ALL stimuli within the resampled clusters,
    and compute the MEAN PAIRED DIFFERENCE inside each resampled pool.

    Use only when `a` and `b` are paired by stimulus index (same query,
    different target). For unpaired comparisons (a from one population,
    b from another) use `two_sample_cluster_bootstrap` instead.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    cluster_ids = np.asarray(cluster_ids)
    unique = np.array(sorted(set(cluster_ids.tolist())))
    idx_by_cluster = {c: np.where(cluster_ids == c)[0] for c in unique}
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        clusters = rng.choice(len(unique), len(unique), replace=True)
        idx = np.concatenate([idx_by_cluster[unique[c]] for c in clusters])
        boots[i] = d[idx].mean()
    raw_p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "mean_diff": float(d.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p_cluster_bootstrap": float(max(raw_p, 1.0 / n_boot)),
        "n_boot": int(n_boot),
        "n_clusters": int(len(unique)),
    }


def welch_two_sample(a: List[float], b: List[float]) -> Dict[str, float]:
    """Welch's unpaired two-sample t-test (unequal variances).

    Use when `a` and `b` are independent samples from different
    populations with no meaningful one-to-one pairing (e.g. distortion
    stimuli vs factual-control stimuli: no shared user prompt, so
    pair-wise subtraction is arbitrary).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {
            "na": na, "nb": nb, "mean_a": float(a.mean()) if na else float("nan"),
            "mean_b": float(b.mean()) if nb else float("nan"),
            "mean_diff": float("nan"), "t": float("nan"), "df": float("nan"),
            "p_t": float("nan"), "cohens_d": float("nan"),
        }
    va, vb = a.var(ddof=1), b.var(ddof=1)
    mean_diff = float(a.mean() - b.mean())
    se = float(np.sqrt(va / na + vb / nb))
    if se == 0:
        return {
            "na": na, "nb": nb, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": mean_diff, "t": float("nan"), "df": float("nan"),
            "p_t": float("nan"), "cohens_d": float("nan"),
        }
    t = mean_diff / se
    # Welch-Satterthwaite df
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    p = _t_sf_two_sided(t, int(round(df)))
    # Pooled-SD Cohen's d (Hedges g without the small-sample correction)
    pooled_sd = float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)))
    d = mean_diff / pooled_sd if pooled_sd > 0 else float("nan")
    return {
        "na": int(na), "nb": int(nb),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": mean_diff, "se": se,
        "t": float(t), "df": float(df), "p_t": float(p),
        "cohens_d": float(d),
    }


def two_sample_cluster_bootstrap(
    a: List[float], b: List[float], cluster_ids_a: List,
    cluster_ids_b: List = None,
    n_boot: int = 5000, seed: int = SEED,
) -> Dict[str, float]:
    """Two-sample bootstrap with optional clustering on BOTH sides.

    If `cluster_ids_b` is provided, `b` is resampled by cluster as well
    (fix R3-b: factual stimuli have subcategory labels too, so treating
    them as i.i.d. understates the variance contributed by within-cluster
    correlation). If `cluster_ids_b` is None, `b` is resampled i.i.d.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cluster_ids_a = np.asarray(cluster_ids_a)
    unique_a = np.array(sorted(set(cluster_ids_a.tolist())))
    idx_by_cluster_a = {c: np.where(cluster_ids_a == c)[0] for c in unique_a}

    use_b_clusters = cluster_ids_b is not None
    if use_b_clusters:
        cluster_ids_b = np.asarray(cluster_ids_b)
        unique_b = np.array(sorted(set(cluster_ids_b.tolist())))
        idx_by_cluster_b = {c: np.where(cluster_ids_b == c)[0] for c in unique_b}

    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    nb = len(b)
    for i in range(n_boot):
        clusters_a = rng.choice(len(unique_a), len(unique_a), replace=True)
        idx_a = np.concatenate([idx_by_cluster_a[unique_a[c]] for c in clusters_a])
        if use_b_clusters:
            clusters_b = rng.choice(len(unique_b), len(unique_b), replace=True)
            idx_b = np.concatenate([idx_by_cluster_b[unique_b[c]] for c in clusters_b])
        else:
            idx_b = rng.choice(nb, nb, replace=True)
        boots[i] = a[idx_a].mean() - b[idx_b].mean()
    raw_p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p_cluster_bootstrap": float(max(raw_p, 1.0 / n_boot)),
        "n_boot": int(n_boot),
        "n_clusters_a": int(len(unique_a)),
        "n_clusters_b": int(len(unique_b)) if use_b_clusters else None,
        "n_b": int(nb),
    }


def holm_bonferroni(ps: List[float]) -> List[float]:
    """Holm step-down adjusted p-values. NaN passes through unchanged."""
    ps = np.asarray(ps, dtype=float)
    valid = np.isfinite(ps)
    out = ps.copy()
    v = ps[valid]
    m = len(v)
    if m == 0:
        return out.tolist()
    order = np.argsort(v)
    adj = np.empty_like(v)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = v[idx] * (m - rank)
        running_max = max(running_max, raw)
        adj[idx] = min(running_max, 1.0)
    out[valid] = adj
    return out.tolist()


def bh_fdr(ps: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted p-values. NaN passes through."""
    ps = np.asarray(ps, dtype=float)
    valid = np.isfinite(ps)
    out = ps.copy()
    v = ps[valid]
    m = len(v)
    if m == 0:
        return out.tolist()
    order = np.argsort(v)
    adj = np.empty_like(v)
    prev = 1.0
    for rank_from_top in range(m):
        rank = m - rank_from_top
        idx = order[rank - 1]
        raw = v[idx] * m / rank
        prev = min(prev, raw)
        adj[idx] = min(prev, 1.0)
    out[valid] = adj
    return out.tolist()


# ---------------------------------------------------------------------------
# G3: mechanism-only steering via residual-stream gradient-contrast PC


def mechanism_steering_direction(
    records: List[dict], layer: int,
) -> torch.Tensor:
    """Top right-singular vector of Δg = (G_T - G_S) at `layer`,
    where each row is the mean-pooled residual-stream gradient of a stimulus.

    Returns unit-norm d_model vector.
    """
    rows = []
    for rec in records:
        gT = rec["T"]["res_g"][layer].mean(0)  # (d_model,)
        gS = rec["S"]["res_g"][layer].mean(0)
        rows.append((gT - gS).numpy())
    M = np.stack(rows).astype(np.float64)  # (N, d_model)
    # Top right-singular vector of M: one SVD.
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v = Vt[0]
    # SVD returns v up to sign; fix so v aligns with the mean row of M,
    # i.e. the "average" gradient contrast direction points the same way.
    mean_row = M.mean(axis=0)
    if float(mean_row @ v) < 0:
        v = -v
    v = torch.from_numpy(v).float()
    return F.normalize(v, dim=0)


# ---------------------------------------------------------------------------
# G4: consensus sharpening across subtypes


def consensus_sharpening(
    records: List[dict], stim: List[dict], layers: List[int],
    variant: str = "pos",
) -> Dict:
    """For each subtype c compute its mean per-layer Δ_rank_ratio profile
       (role T minus role S), stack into a (n_cat, L) matrix, return its
       stable rank and singular spectrum.

    Interpretation: low stable rank = all 12 subtypes share a single
    cross-layer signature of therapeutic-capacity gap. High stable rank
    = subtype-specific gaps with little shared structure.
    """
    by_cat: Dict[str, List[int]] = {}
    for i, s in enumerate(stim):
        by_cat.setdefault(s["subcategory"], []).append(i)

    cats = sorted(by_cat)
    rows = []
    key = f"rank_ratio_{variant}"
    for c in cats:
        ix = by_cat[c]
        diffs_per_layer = []
        for l in layers:
            rrs_T = [records[i]["T"]["ranks"][l][key] for i in ix]
            rrs_S = [records[i]["S"]["ranks"][l][key] for i in ix]
            diffs_per_layer.append(float(np.mean(rrs_T) - np.mean(rrs_S)))
        rows.append(diffs_per_layer)
    M = np.asarray(rows, dtype=np.float64)  # (n_cat, L)
    # Centre per-column (remove the population mean at each layer) so the
    # stable rank measures subtype-level variation around the mean profile.
    M_centred = M - M.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(M_centred, full_matrices=False)
    lam = S ** 2
    srank = stable_ranks(torch.from_numpy(lam))
    return {
        "subcategories": cats,
        "layers": layers,
        "profile_matrix": M.tolist(),
        "centred_singular_values": S.tolist(),
        "stable_rank_pre": srank["pre"],
        "stable_rank_pos": srank["pos"],
        "mean_profile": M.mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main pipeline


def run(args):
    set_seeds()
    device = select_device(args.device)
    print(f"Device: {device}")

    model_id = MODEL_IDS[args.model]
    print(f"Loading {model_id} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s; "
          f"n_layers={model.config.num_hidden_layers}, "
          f"d_model={model.config.hidden_size}, "
          f"d_ff={model.config.intermediate_size}")

    n_layers = model.config.num_hidden_layers
    all_layers = pick_layers(n_layers, n_keep=args.n_layers if args.n_layers > 0 else None)
    # MLP layers: where we run the full GRADE projected rank analysis.
    mlp_layers = all_layers
    # Residual layers for G3 steering: only a few; cheaper.
    res_layers = [all_layers[len(all_layers) // 2]]
    target_layer = res_layers[0]
    print(f"MLP layers: {len(mlp_layers)}; residual (steering) layer: {target_layer}")

    # ------------------------------------------------------------------
    # Stimuli
    raw_dist = load_json(STIM_DIR / "cognitive_distortions.json")
    raw_fact = load_json(STIM_DIR / "v2_factual_control.json")

    dist_stim = stratified_sample(raw_dist, args.n_per_cat if args.n_per_cat > 0 else None)
    fact_stim = sorted(raw_fact, key=lambda x: x["id"])[: len(dist_stim)]
    n_subcats = len(set(s["subcategory"] for s in dist_stim))
    print(f"Distortion stimuli: {len(dist_stim)} (across {n_subcats} subcats)")
    print(f"Factual control stimuli: {len(fact_stim)}")

    # ------------------------------------------------------------------
    # Extract GRADE data: one fwd+bwd per (stim, role) pair.
    print("\n[extract] distortion: T=therapeutic, S=sycophantic")
    dist_records = extract_paired_grad_data(
        model, tokenizer, dist_stim,
        "therapeutic_completion", "sycophantic_completion",
        mlp_layers, res_layers, desc="dist grad",
    )
    print("\n[extract] factual control: T=therapeutic, S=sycophantic "
          "(used as knowledge-baseline reference for G5)")
    fact_records = extract_paired_grad_data(
        model, tokenizer, fact_stim,
        "therapeutic_completion", "sycophantic_completion",
        mlp_layers, [], desc="fact grad",
    )

    # ------------------------------------------------------------------
    # G1 + G5: per-layer rank ratios
    print("\n[G1/G5] per-layer rank ratios (therapeutic vs sycophantic)")
    rr_T = per_stim_rank_ratios(dist_records, mlp_layers, role="T")
    rr_S = per_stim_rank_ratios(dist_records, mlp_layers, role="S")
    rr_T_fact = per_stim_rank_ratios(fact_records, mlp_layers, role="T")

    # Cluster ids for cluster-bootstrap (within-subtype stimuli are not i.i.d.).
    subtype_ids = [s["subcategory"] for s in dist_stim]
    fact_subtype_ids = [s.get("subcategory", s.get("category", str(s["id"]))) for s in fact_stim]

    g1_by_layer = {}
    raw_ps_T_vs_S = []
    for l in mlp_layers:
        pos_T = [r["rank_ratio_pos"] for r in rr_T[l]]
        pos_S = [r["rank_ratio_pos"] for r in rr_S[l]]
        pre_T = [r["rank_ratio_pre"] for r in rr_T[l]]
        pre_S = [r["rank_ratio_pre"] for r in rr_S[l]]
        pos_Tf = [r["rank_ratio_pos"] for r in rr_T_fact[l]]
        pt = paired_tests(pos_T, pos_S)
        # T_dist vs T_factual is unpaired (different source datasets, no
        # meaningful stimulus-level correspondence). Welch + two-sample
        # cluster bootstrap replace the erroneous list-position paired-t.
        tf_welch = welch_two_sample(pos_T, pos_Tf)
        tf_cb = two_sample_cluster_bootstrap(pos_T, pos_Tf, subtype_ids, fact_subtype_ids)
        cb = cluster_bootstrap_ci_mean_diff(pos_T, pos_S, subtype_ids, n_boot=2000)
        g1_by_layer[l] = {
            "pos": {
                "mean_T_dist": float(np.mean(pos_T)),
                "mean_S_dist": float(np.mean(pos_S)),
                "mean_T_factual": float(np.mean(pos_Tf)),
                "paired_T_vs_S_dist": pt,
                "welch_T_dist_vs_T_factual": tf_welch,
                "two_sample_cluster_bootstrap_T_dist_vs_T_factual": tf_cb,
                "cluster_bootstrap_T_vs_S_dist": cb,
                "bootstrap_T_vs_S_dist": bootstrap_ci_mean_diff(pos_T, pos_S),
            },
            "pre": {
                "mean_T_dist": float(np.mean(pre_T)),
                "mean_S_dist": float(np.mean(pre_S)),
                "paired_T_vs_S_dist": paired_tests(pre_T, pre_S),
            },
            "per_stim_pos_T": pos_T,
            "per_stim_pos_S": pos_S,
            "per_stim_pos_T_factual": pos_Tf,
        }
        raw_ps_T_vs_S.append(pt["p_t"])
        print(f"  L{l:2d}: mean_T={g1_by_layer[l]['pos']['mean_T_dist']:.3f}  "
              f"mean_S={g1_by_layer[l]['pos']['mean_S_dist']:.3f}  "
              f"Δ={pt['mean_diff']:+.3f} (t={pt['t']:+.2f} df={pt['n']-1}, "
              f"p_t={pt['p_t']:.3f}, p_sign={pt['p_sign']:.3f}, "
              f"d_z={pt['cohens_dz']:+.2f}, cluster95=[{cb['ci_lo']:+.3f},{cb['ci_hi']:+.3f}])")

    # Holm & BH correction across the G1 per-layer T-vs-S tests.
    holm_ps = holm_bonferroni(raw_ps_T_vs_S)
    bh_ps = bh_fdr(raw_ps_T_vs_S)
    for l, hp, bp in zip(mlp_layers, holm_ps, bh_ps):
        g1_by_layer[l]["pos"]["paired_T_vs_S_dist"]["p_t_holm"] = hp
        g1_by_layer[l]["pos"]["paired_T_vs_S_dist"]["p_t_bh"] = bp
    print("  multi-comparison (Holm / BH across " + str(len(mlp_layers)) + " layers):")
    for l, hp, bp in zip(mlp_layers, holm_ps, bh_ps):
        print(f"    L{l:2d}: Holm p={hp:.3f}  BH p={bp:.3f}")

    # G5 capacity summary at mid layer and mean-over-layers.
    print("\n[G5] capacity summary (mean across layers):")
    all_T = np.array([[rr_T[l][i]["rank_ratio_pos"] for i in range(len(rr_T[l]))]
                      for l in mlp_layers])  # (L, N)
    all_S = np.array([[rr_S[l][i]["rank_ratio_pos"] for i in range(len(rr_S[l]))]
                      for l in mlp_layers])
    all_Tf = np.array([[rr_T_fact[l][i]["rank_ratio_pos"] for i in range(len(rr_T_fact[l]))]
                       for l in mlp_layers])
    per_stim_mean_T = all_T.mean(axis=0)  # (N,)
    per_stim_mean_S = all_S.mean(axis=0)
    per_stim_mean_Tf = all_Tf.mean(axis=0)
    # Capacity proxy C = 1 / rank_ratio_pos; higher = more capacity.
    cap_T = 1.0 / np.clip(per_stim_mean_T, 1e-6, None)
    cap_S = 1.0 / np.clip(per_stim_mean_S, 1e-6, None)
    cap_Tf = 1.0 / np.clip(per_stim_mean_Tf, 1e-6, None)
    # Power calc for the observed effect sizes. For paired contrasts (d_z)
    # the required n is ((z_a+z_b)/d_z)^2. For unpaired two-sample contrasts
    # (Cohen's d, pooled) the required PER-GROUP n is 2× that. We expose
    # the `paired` flag so call sites can pick the right one (fix R3-a).
    def _power_n(d: float, paired: bool = True,
                 alpha: float = 0.05, power: float = 0.8) -> int:
        if not np.isfinite(d) or abs(d) < 1e-6:
            return -1
        z_a = 1.96      # two-sided α = 0.05
        z_b = 0.842     # Φ^{-1}(0.8)
        n_paired = ((z_a + z_b) / abs(d)) ** 2
        if paired:
            return int(np.ceil(n_paired))
        return int(np.ceil(2 * n_paired))   # per-group n for two-sample

    pt_ts = paired_tests(cap_T.tolist(), cap_S.tolist())
    tf_welch = welch_two_sample(cap_T.tolist(), cap_Tf.tolist())  # unpaired
    cb_ts = cluster_bootstrap_ci_mean_diff(cap_T.tolist(), cap_S.tolist(), subtype_ids, n_boot=5000)
    cb_tf = two_sample_cluster_bootstrap(
        cap_T.tolist(), cap_Tf.tolist(), subtype_ids, fact_subtype_ids, n_boot=5000)
    # Power calc for the T_vs_T_factual contrast uses Cohen's d (pooled),
    # not d_z (paired): same z_a/z_b approximation.
    g5 = {
        "capacity_mean_therapeutic_dist": float(cap_T.mean()),
        "capacity_mean_sycophantic_dist": float(cap_S.mean()),
        "capacity_mean_therapeutic_factual": float(cap_Tf.mean()),
        "paired_T_vs_S_dist": pt_ts,
        "welch_T_dist_vs_T_factual": tf_welch,
        "cluster_bootstrap_T_vs_S_dist": cb_ts,
        "two_sample_cluster_bootstrap_T_dist_vs_T_factual": cb_tf,
        "bootstrap_T_vs_S_dist": bootstrap_ci_mean_diff(cap_T.tolist(), cap_S.tolist()),
        "power_n_for_T_vs_S": _power_n(pt_ts.get("cohens_dz", float("nan")), paired=True),
        "power_n_for_T_vs_T_factual": _power_n(
            tf_welch.get("cohens_d", float("nan")), paired=False),
        "per_stim_capacity_T_dist": cap_T.tolist(),
        "per_stim_capacity_S_dist": cap_S.tolist(),
        "per_stim_capacity_T_factual": cap_Tf.tolist(),
    }
    print(f"  C_ther(dist)   = {g5['capacity_mean_therapeutic_dist']:.4f}")
    print(f"  C_syc(dist)    = {g5['capacity_mean_sycophantic_dist']:.4f}")
    print(f"  C_ther(factual)= {g5['capacity_mean_therapeutic_factual']:.4f}")
    print(f"  paired T vs S  Δ={pt_ts['mean_diff']:+.4f}  t={pt_ts['t']:+.2f} df={pt_ts['n']-1}  "
          f"p_t={pt_ts['p_t']:.3f}  p_sign={pt_ts['p_sign']:.3f}  d_z={pt_ts['cohens_dz']:+.2f}  "
          f"cluster95=[{cb_ts['ci_lo']:+.4f},{cb_ts['ci_hi']:+.4f}]  "
          f"n_for_80%_power≈{g5['power_n_for_T_vs_S']}")
    print(f"  T dist vs fact (Welch, UNPAIRED) "
          f"Δ={tf_welch['mean_diff']:+.4f}  t={tf_welch['t']:+.2f} "
          f"df≈{tf_welch['df']:.1f}  p_t={tf_welch['p_t']:.3f}  d={tf_welch['cohens_d']:+.2f}  "
          f"2s_cluster95=[{cb_tf['ci_lo']:+.4f},{cb_tf['ci_hi']:+.4f}]  "
          f"p_2s_boot={cb_tf['p_cluster_bootstrap']:.3f}  "
          f"n_for_80%_power≈{g5['power_n_for_T_vs_T_factual']}")

    # ------------------------------------------------------------------
    # G3: mechanism steering (v* fit on fit-half ONLY; evaluated on held-out half)
    print(f"\n[G3] mechanism steering direction at L{target_layer}")
    # FIT set: first half of distortion stimuli.
    # EVAL (intervention) set: second half, up to n_intervene.
    n_fit = len(dist_stim) // 2
    fit_records = dist_records[:n_fit]
    fit_stim = dist_stim[:n_fit]
    inter_stim = dist_stim[n_fit:][: args.n_intervene] if args.n_intervene > 0 else dist_stim[n_fit:]
    print(f"  fit on {len(fit_records)} stimuli; intervene on {len(inter_stim)} held-out stimuli")

    v_star = mechanism_steering_direction(fit_records, target_layer)
    # Baseline direction d from repo: mean activation contrast (syc - ther)
    # ALSO fit on the first-half stimuli only, matched out-of-sample comparison.
    syc_fit_acts, ther_fit_acts = extract_paired(
        model, tokenizer, fit_stim,
        "sycophantic_completion", "therapeutic_completion",
        [target_layer], desc="acts (fit-half only)",
    )
    d_baseline = contrastive_direction(syc_fit_acts, ther_fit_acts)[target_layer]
    cos_vstar_d = float(F.cosine_similarity(v_star.unsqueeze(0), d_baseline.unsqueeze(0)).item())
    print(f"  cos(v*, d_baseline) = {cos_vstar_d:+.3f}")

    pairs = [
        ("therapeutic_completion", "sycophantic_completion"),  # E5
        ("therapeutic_completion", "cold_completion"),         # E6
    ]
    pair_keys = {pairs[0]: "E5_ther_vs_syc", pairs[1]: "E6_ther_vs_cold"}

    # Baseline logprob signals (once, no hook).
    with torch.no_grad():
        baseline_signals = compute_baseline_signals(model, tokenizer, inter_stim, pairs)

    def shifts(direction: torch.Tensor, alpha: float):
        with torch.no_grad():
            sigs = compute_intervention_signals(
                model, tokenizer, inter_stim, target_layer, direction, pairs,
                intervention="negative_steering", alpha=alpha,
            )
        # Per-stimulus shifts for bootstrap CIs.
        per_stim = {
            pair_keys[p]: (np.asarray(sigs[p]) - np.asarray(baseline_signals[p])).tolist()
            for p in pairs
        }
        summaries = {pair_keys[p]: shift_summary(baseline_signals[p], sigs[p]) for p in pairs}
        dE5 = summaries["E5_ther_vs_syc"]["shift_mean"]
        dE6 = summaries["E6_ther_vs_cold"]["shift_mean"]
        summaries["specificity_ratio"] = abs(dE5) / max(abs(dE6), 1e-8)
        summaries["E5_shift"] = dE5
        summaries["E6_shift"] = dE6
        summaries["per_stim"] = per_stim
        return summaries

    def bootstrap_ratio_ci(per_stim: dict, n_boot: int = 2000, seed: int = SEED):
        e5 = np.asarray(per_stim["E5_ther_vs_syc"], dtype=np.float64)
        e6 = np.asarray(per_stim["E6_ther_vs_cold"], dtype=np.float64)
        rng = np.random.RandomState(seed)
        n = len(e5)
        boots_e5 = np.empty(n_boot); boots_e6 = np.empty(n_boot); boots_r = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            m5 = e5[idx].mean(); m6 = e6[idx].mean()
            boots_e5[i] = m5; boots_e6[i] = m6
            boots_r[i] = abs(m5) / max(abs(m6), 1e-8)
        return {
            "E5_shift_ci": [float(np.percentile(boots_e5, 2.5)), float(np.percentile(boots_e5, 97.5))],
            "E6_shift_ci": [float(np.percentile(boots_e6, 2.5)), float(np.percentile(boots_e6, 97.5))],
            "specificity_ratio_ci": [float(np.percentile(boots_r, 2.5)),
                                     float(np.percentile(boots_r, 97.5))],
            "specificity_ratio_median_boot": float(np.median(boots_r)),
            "n_boot": int(n_boot),
        }

    alpha = args.alpha
    print(f"  alpha={alpha}")
    print("  scoring v* ...")
    res_vstar = shifts(v_star, alpha)
    ci_vstar = bootstrap_ratio_ci(res_vstar["per_stim"])
    print(f"    v*         ΔE5={res_vstar['E5_shift']:+.3f} {ci_vstar['E5_shift_ci']}  "
          f"ΔE6={res_vstar['E6_shift']:+.3f} {ci_vstar['E6_shift_ci']}  "
          f"spec={res_vstar['specificity_ratio']:.2f} CI{ci_vstar['specificity_ratio_ci']}")
    print("  scoring d_baseline ...")
    res_dbase = shifts(d_baseline, alpha)
    ci_dbase = bootstrap_ratio_ci(res_dbase["per_stim"])
    print(f"    d_baseline ΔE5={res_dbase['E5_shift']:+.3f} {ci_dbase['E5_shift_ci']}  "
          f"ΔE6={res_dbase['E6_shift']:+.3f} {ci_dbase['E6_shift_ci']}  "
          f"spec={res_dbase['specificity_ratio']:.2f} CI{ci_dbase['specificity_ratio_ci']}")

    # Random-direction null (I2).
    n_random = args.n_random
    print(f"  scoring {n_random} random directions ...")
    rng_r = np.random.RandomState(SEED + 7)
    d_model = v_star.shape[0]
    rand_specs = []
    rand_e5s = []
    rand_e6s = []
    for k in range(n_random):
        rv = torch.from_numpy(rng_r.randn(d_model).astype(np.float32))
        rv = F.normalize(rv, dim=0)
        r = shifts(rv, alpha)
        rand_specs.append(r["specificity_ratio"])
        rand_e5s.append(r["E5_shift"])
        rand_e6s.append(r["E6_shift"])
    rand_specs = np.asarray(rand_specs)
    rand_e5s = np.asarray(rand_e5s)
    rand_e6s = np.asarray(rand_e6s)
    vstar_p_random_spec = float((rand_specs >= res_vstar["specificity_ratio"]).mean())
    vstar_p_random_e5 = float((rand_e5s >= res_vstar["E5_shift"]).mean())
    dbase_p_random_spec = float((rand_specs >= res_dbase["specificity_ratio"]).mean())
    dbase_p_random_e5 = float((rand_e5s >= res_dbase["E5_shift"]).mean())
    print(f"    random:    ΔE5 mean={rand_e5s.mean():+.3f} sd={rand_e5s.std(ddof=1):.3f}  "
          f"ΔE6 mean={rand_e6s.mean():+.3f} sd={rand_e6s.std(ddof=1):.3f}  "
          f"spec mean={rand_specs.mean():.2f}")
    print(f"    v* vs random: p_spec={vstar_p_random_spec:.3f}  p_E5={vstar_p_random_e5:.3f}")
    print(f"    d_baseline vs random: p_spec={dbase_p_random_spec:.3f}  p_E5={dbase_p_random_e5:.3f}")

    # Drop per_stim from the saved summaries (already used for CIs).
    res_vstar_save = {k: v for k, v in res_vstar.items() if k != "per_stim"}
    res_dbase_save = {k: v for k, v in res_dbase.items() if k != "per_stim"}

    g3 = {
        "layer": target_layer,
        "alpha": alpha,
        "n_fit": len(fit_records),
        "n_intervene": len(inter_stim),
        "cos_vstar_d_baseline": cos_vstar_d,
        "v_star": res_vstar_save,
        "v_star_bootstrap_ci": ci_vstar,
        "d_baseline": res_dbase_save,
        "d_baseline_bootstrap_ci": ci_dbase,
        "specificity_ratio_vstar": res_vstar["specificity_ratio"],
        "specificity_ratio_d_baseline": res_dbase["specificity_ratio"],
        "random_null": {
            "n_random": int(n_random),
            "spec_mean": float(rand_specs.mean()),
            "spec_sd": float(rand_specs.std(ddof=1)),
            "E5_mean": float(rand_e5s.mean()),
            "E5_sd": float(rand_e5s.std(ddof=1)),
            "E6_mean": float(rand_e6s.mean()),
            "E6_sd": float(rand_e6s.std(ddof=1)),
            "vstar_p_random_spec": vstar_p_random_spec,
            "vstar_p_random_E5_shift": vstar_p_random_e5,
            "dbase_p_random_spec": dbase_p_random_spec,
            "dbase_p_random_E5_shift": dbase_p_random_e5,
        },
    }

    # ------------------------------------------------------------------
    # G4: consensus sharpening
    print(f"\n[G4] consensus sharpening across subtypes (n_cat >= 2 required)")
    g4 = consensus_sharpening(dist_records, dist_stim, mlp_layers, variant="pos")
    print(f"  subcategories: {len(g4['subcategories'])}")
    print(f"  centred (n_cat × L) stable rank pre = {g4['stable_rank_pre']:.3f}")
    print(f"  centred (n_cat × L) stable rank pos = {g4['stable_rank_pos']:.3f}")
    print(f"  mean Δ-rank-ratio profile: " +
          ", ".join(f"L{l}={m:+.2f}" for l, m in zip(mlp_layers, g4["mean_profile"])))

    # ------------------------------------------------------------------
    # Save and plot
    out = {
        "config": {
            "model_id": model_id,
            "n_layers": n_layers,
            "mlp_layers": mlp_layers,
            "steering_layer": target_layer,
            "n_distortion_stim": len(dist_stim),
            "n_factual_stim": len(fact_stim),
            "n_intervene": len(inter_stim),
            "alpha": alpha,
            "seed": SEED,
            "variant": "rank_ratio_pos",
        },
        "G1_rank_ratios_per_layer": g1_by_layer,
        "G5_capacity_summary": g5,
        "G3_mechanism_steering": g3,
        "G4_consensus_sharpening": g4,
    }
    save_path = RESULTS_DIR / "grade_results.json"
    save_json(out, save_path)
    print(f"\nSaved {save_path}")

    make_grade_figures(out, mlp_layers, target_layer)
    print(f"Saved figures to {FIGURES_DIR}")


def make_grade_figures(res: dict, layers: List[int], target_layer: int) -> None:
    g1 = res["G1_rank_ratios_per_layer"]
    # Fig A: per-layer mean rank ratio (T vs S) on distortion vs factual
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    mT = [g1[str(l)]["pos"]["mean_T_dist"] if str(l) in g1 else g1[l]["pos"]["mean_T_dist"] for l in layers]
    mS = [g1[str(l)]["pos"]["mean_S_dist"] if str(l) in g1 else g1[l]["pos"]["mean_S_dist"] for l in layers]
    mTf = [g1[str(l)]["pos"]["mean_T_factual"] if str(l) in g1 else g1[l]["pos"]["mean_T_factual"] for l in layers]
    ax.plot(layers, mT, "o-", color=COLOR["ther"], label="therapeutic (distortion)")
    ax.plot(layers, mS, "s-", color=COLOR["syc"], label="sycophantic (distortion)")
    ax.plot(layers, mTf, "^--", color=COLOR["fact"], label="therapeutic (factual ctrl)")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"mean rank ratio $\mathrm{srank}_{\mathrm{pos}}(C_g)/\mathrm{srank}_{\mathrm{pos}}(C_h)$")
    ax.set_title("G1: Per-layer knowledge-gap signal (lower = more activated capacity)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grade_g1_rank_ratio_by_layer.png")
    plt.close(fig)

    # Fig B: capacity histogram
    g5 = res["G5_capacity_summary"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cT = np.asarray(g5["per_stim_capacity_T_dist"])
    cS = np.asarray(g5["per_stim_capacity_S_dist"])
    cTf = np.asarray(g5["per_stim_capacity_T_factual"])
    bins = np.linspace(0, max(cT.max(), cS.max(), cTf.max()) * 1.05, 24)
    ax.hist(cT, bins=bins, alpha=0.5, color=COLOR["ther"], label=f"C_ther(dist)  μ={cT.mean():.3f}")
    ax.hist(cS, bins=bins, alpha=0.5, color=COLOR["syc"], label=f"C_syc(dist)   μ={cS.mean():.3f}")
    ax.hist(cTf, bins=bins, alpha=0.5, color=COLOR["fact"], label=f"C_ther(fact)  μ={cTf.mean():.3f}")
    ax.set_xlabel("capacity C = 1 / mean_l RankRatio_pos(l)")
    ax.set_ylabel("count")
    ax.set_title("G5: Per-stimulus GRADE capacity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grade_g5_capacity_hist.png")
    plt.close(fig)

    # Fig C: G3 specificity
    g3 = res["G3_mechanism_steering"]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    labels = ["d_baseline\n(activation)", "v*\n(gradient PC)"]
    specs = [g3["specificity_ratio_d_baseline"], g3["specificity_ratio_vstar"]]
    bars = ax.bar(labels, specs, color=[COLOR["neutral"], COLOR["cap"]], alpha=0.85)
    for b, v in zip(bars, specs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("specificity ratio |ΔE5| / |ΔE6|")
    ax.set_title(f"G3: Mechanism vs baseline steering (L{g3['layer']}, α={g3['alpha']})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grade_g3_specificity.png")
    plt.close(fig)

    # Fig D: G4 consensus
    g4 = res["G4_consensus_sharpening"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    prof = np.asarray(g4["profile_matrix"])
    for i, c in enumerate(g4["subcategories"]):
        ax.plot(layers, prof[i], alpha=0.6, lw=0.9)
    ax.plot(layers, g4["mean_profile"], "k-", lw=2, label="mean over subtypes")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ rank ratio  (T − S)")
    ax.set_title(f"G4: Per-subtype capacity-gap profile. "
                 f"srank_pos(centred)={g4['stable_rank_pos']:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grade_g4_consensus.png")
    plt.close(fig)


def parse_args(argv: List[str] = None):
    """Parse CLI arguments. If `argv` is None, uses sys.argv (CLI path);
    if a list is supplied, parses that (notebook path). This lets callers
    in the Colab notebook build a `Namespace` from the same source of
    truth as the CLI — so adding a new argparse flag does not silently
    break the notebook with AttributeError.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), default="1b")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-per-cat", type=int, default=3,
                    help="distortion stimuli per subcategory; 0=all")
    ap.add_argument("--n-intervene", type=int, default=16,
                    help="held-out stimuli for G3 steering; 0=all held-out")
    ap.add_argument("--n-random", type=int, default=20,
                    help="random-direction controls for G3 null")
    ap.add_argument("--n-layers", type=int, default=8,
                    help="number of layers to sample; 0=all")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.n_per_cat = 1
        args.n_intervene = 6
        args.n_layers = 5
        args.n_random = 5
    return args


if __name__ == "__main__":
    run(parse_args())
