"""
Mechanistic Interpretability of Cognitive Distortion Validation in LLMs.

Self-contained reference implementation. Runs all six experiments end-to-end
on OLMo-2 1B for local validation; the same script generalizes to OLMo-3 7B
Instruct for the final Colab run by changing MODEL_ID.

Experiments:
  E1  Identify a "distortion-acceptance" direction via contrastive probing
      between sycophantic and therapeutic completions on cognitive-distortion
      stimuli; report per-layer AUC and a permutation test against a label-
      shuffled null at a pre-registered layer.
  E2  Disentangle from "empathic warmth" (therapeutic vs cold completions);
      report cosine similarity and Gram-Schmidt variance decomposition.
  E3  Localize the direction across layers using projection magnitude on
      held-out stimuli.
  E4  Cross-distortion generalization via leave-one-out: train probe on 11
      distortion types, test on the held-out type.
  E5  Causal intervention: at the pre-registered middle layer, both
      projection-ablation and negative-steering hooks are applied; we
      measure the shift in log P(therapeutic) - log P(sycophantic) at the
      assistant response position, against a random-direction control.
  E6  Therapeutic specificity: same intervention, measure shift in
      log P(therapeutic) - log P(cold). A specific intervention should
      reduce the sycophancy signal (E5) without degrading the warmth signal
      (E6); we report the specificity ratio |Δ_E5| / |Δ_E6|.
  E7  Per-subtype geometry: for each of the 12 cognitive-distortion
      subtypes, compute its own contrastive direction at every sampled
      layer.
  E8  Subspace dimensionality: SVD on the (12 × d_model) stacked subtype-
      direction matrix at every layer; report singular values, variance
      explained per principal component, and the participation ratio (an
      effective-dimensionality measure).
  E9  Pairwise distortion-direction similarity: the 12 × 12 cosine matrix
      among per-subtype directions at every layer; we cluster distortions
      by similarity and report each subtype's variance retained when
      projected onto the top-k principal subspace.

Usage:
    python reference.py                      # OLMo-2 1B local
    python reference.py --model 7b           # OLMo-3 7B Instruct (Colab)
    python reference.py --quick              # 30 stimuli for smoke test

Outputs:
    results/results.json
    figures/*.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration

SEED = 42
ROOT = Path(__file__).parent
STIM_DIR = ROOT / "stimuli"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_IDS = {
    "1b": "allenai/OLMo-2-0425-1B-Instruct",
    "7b": "allenai/Olmo-3-7B-Instruct",
}

COLOR = {
    "dist": "#c0392b",
    "warmth": "#2980b9",
    "factual": "#27ae60",
    "neutral": "#7f8c8d",
    "purple": "#8e44ad",
    "orange": "#e67e22",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Helpers


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(model) -> torch.device:
    return next(model.parameters()).device


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def select_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    def _coerce(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, dict):
            return {str(k): _coerce(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_coerce(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(_coerce(obj), f, indent=2)


def format_prompt(tokenizer, user_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return user_text


# ---------------------------------------------------------------------------
# Activation extraction


@torch.no_grad()
def _hidden_states(model, input_ids: torch.Tensor, layers: List[int]) -> Dict[int, torch.Tensor]:
    hidden: Dict[int, torch.Tensor] = {}
    hooks = []
    targets = set(layers)

    def make_hook(idx: int):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden[idx] = h.detach().cpu().float().squeeze(0)
        return fn

    for i in targets:
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))
    try:
        model(input_ids.to(get_device(model)))
    finally:
        for h in hooks:
            h.remove()
    return hidden


def _completion_acts(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Mean-pool the residual stream over the entire completion at each layer.

    No truncation: we want the full content of each completion to contribute
    to the per-stimulus activation. Mean-pooling implicitly handles length
    differences (sycophantic 79 / therapeutic 89 / cold 78 mean words) by
    averaging per-token; capping length would drop late-completion content
    that carries the explicit therapeutic / cold framing.
    """
    formatted = format_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(formatted, return_tensors="pt")
    full_ids = tokenizer.encode(formatted + completion, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]
    # Prefix invariant: the prompt-only encoding must be a strict prefix of
    # the prompt+completion encoding. If a tokenizer ever retokenizes across
    # the boundary, prompt_len-based slicing would silently corrupt every
    # downstream activation and log-prob.
    assert prompt_len <= full_ids.shape[1] and (
        full_ids[0, :prompt_len].tolist() == prompt_ids[0].tolist()
    ), "tokenizer is retokenizing across the prompt/completion boundary"

    hidden = _hidden_states(model, full_ids, layers)
    pooled: Dict[int, torch.Tensor] = {}
    for layer, h in hidden.items():
        comp = h[prompt_len:]
        if len(comp) == 0:
            comp = h[-1:]
        pooled[layer] = comp.mean(0)
    return pooled


def extract_paired(
    model,
    tokenizer,
    stimuli: List[dict],
    pos_key: str,
    neg_key: str,
    layers: List[int],
    desc: str = "Extracting",
) -> Tuple[List[Dict[int, torch.Tensor]], List[Dict[int, torch.Tensor]]]:
    pos_list, neg_list = [], []
    for i, s in enumerate(tqdm(stimuli, desc=desc)):
        pos_list.append(_completion_acts(model, tokenizer, s["user_prompt"], s[pos_key], layers))
        neg_list.append(_completion_acts(model, tokenizer, s["user_prompt"], s[neg_key], layers))
        if (i + 1) % 20 == 0:
            cleanup()
    return pos_list, neg_list


# ---------------------------------------------------------------------------
# Direction computation, decomposition, probing


def contrastive_direction(pos_acts, neg_acts) -> Dict[int, torch.Tensor]:
    layers = sorted(pos_acts[0].keys())
    out = {}
    for l in layers:
        pos = torch.stack([a[l] for a in pos_acts])
        neg = torch.stack([a[l] for a in neg_acts])
        diff = pos.mean(0) - neg.mean(0)
        out[l] = F.normalize(diff, dim=0)
    return out


def cosine_by_layer(da, db):
    layers = sorted(set(da) & set(db))
    return {l: F.cosine_similarity(da[l].unsqueeze(0), db[l].unsqueeze(0)).item() for l in layers}


def project(acts_list, directions):
    out = {l: [] for l in directions}
    for a in acts_list:
        for l in directions:
            out[l].append((a[l] @ directions[l]).item())
    return out


def within_domain_probe(pos_acts, neg_acts, layers: List[int], cv: int = 5):
    out = {}
    for l in layers:
        X = np.concatenate([
            np.stack([a[l].numpy() for a in pos_acts]),
            np.stack([a[l].numpy() for a in neg_acts]),
        ])
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        n_cv = min(cv, int(min(y.sum(), (1 - y).sum())))
        if n_cv < 2:
            out[l] = {"acc_mean": float("nan"), "auc_mean": float("nan")}
            continue
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=SEED)
        accs, aucs = [], []
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            prob = clf.predict_proba(X[te])[:, 1]
            accs.append(accuracy_score(y[te], pred))
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except ValueError:
                aucs.append(float("nan"))
        out[l] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
        }
    return out


def cross_domain_probe(src_pos, src_neg, tgt_pos, tgt_neg, layers):
    out = {}
    for l in layers:
        Xtr = np.concatenate([
            np.stack([a[l].numpy() for a in src_pos]),
            np.stack([a[l].numpy() for a in src_neg]),
        ])
        ytr = np.concatenate([np.ones(len(src_pos)), np.zeros(len(src_neg))])
        Xte = np.concatenate([
            np.stack([a[l].numpy() for a in tgt_pos]),
            np.stack([a[l].numpy() for a in tgt_neg]),
        ])
        yte = np.concatenate([np.ones(len(tgt_pos)), np.zeros(len(tgt_neg))])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        try:
            auc = float(roc_auc_score(yte, prob))
        except ValueError:
            auc = float("nan")
        out[l] = {"acc": float(accuracy_score(yte, clf.predict(Xte))), "auc": auc}
    return out


def per_subtype_directions(stim: List[dict], pos_acts, neg_acts,
                           layers: List[int]) -> Dict[str, Dict[int, torch.Tensor]]:
    """Compute the contrastive direction separately for each subcategory.

    Returns {subcategory: {layer: unit-direction}}.
    """
    by_cat: Dict[str, List[int]] = {}
    for i, s in enumerate(stim):
        by_cat.setdefault(s["subcategory"], []).append(i)
    out = {}
    for cat, ix in by_cat.items():
        sp = [pos_acts[i] for i in ix]
        sn = [neg_acts[i] for i in ix]
        out[cat] = contrastive_direction(sp, sn)
    return out


def subtype_geometry(subtype_dirs: Dict[str, Dict[int, torch.Tensor]], layer: int,
                     ks: List[int] = (1, 2, 3, 5)) -> dict:
    """Geometry of the per-subtype direction set at one layer.

    Uses SVD on the 12 × d_model matrix of stacked unit-norm directions to:
      - report the singular value spectrum (each subtype contributes one row),
      - compute the participation ratio (sum(s^2)^2 / sum(s^4)) as an
        effective dimensionality measure,
      - compute the variance fraction of each principal component,
      - report the fraction of each subtype direction's variance retained when
        projected onto the top-k principal subspace,
      - return the full pairwise cosine matrix among subtypes.
    """
    cats = sorted(subtype_dirs)
    # float64 to avoid fp32 overflow / nan in d_model x n_cat matmul on MPS;
    # directions arrive as fp32 from the model but are normalized so the
    # cast is safe and inexpensive at this size.
    D = np.stack([subtype_dirs[c][layer].numpy() for c in cats]).astype(np.float64)
    # Each row is unit norm; squared norm = 1 → trace(D D^T) = n_cat.
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    s2 = S ** 2
    var_frac = (s2 / s2.sum()).tolist()
    cum_var = np.cumsum(var_frac).tolist()
    pr = float(s2.sum() ** 2 / (s2 ** 2).sum())
    cos_mat = (D @ D.T).tolist()  # rows are unit norm

    proj_var = {}
    for k in ks:
        if k > Vt.shape[0]:
            continue
        Pk = Vt[:k]  # (k, d_model)
        coefs = D @ Pk.T  # (n_cat, k)
        retained = (coefs ** 2).sum(axis=1)  # max=1 since |D[i]|=1
        proj_var[k] = {c: float(retained[i]) for i, c in enumerate(cats)}
    return {
        "subcategories": cats,
        "singular_values": S.tolist(),
        "var_fraction": var_frac,
        "cumulative_var_fraction": cum_var,
        "participation_ratio": pr,
        "pairwise_cosine": cos_mat,
        "subspace_variance_retained_top_k": proj_var,
    }


def decompose_direction(target: torch.Tensor, components: Dict[str, torch.Tensor]):
    """Gram-Schmidt unique-variance decomposition of a target direction."""
    total = (target.norm() ** 2).item()
    residual = target.clone()
    used: List[torch.Tensor] = []
    out = {"unique_ve": {}, "raw_proj": {}}
    for name, comp in components.items():
        cn = F.normalize(comp, dim=0)
        out["raw_proj"][name] = (target @ cn).item()
    for name, comp in components.items():
        cn = F.normalize(comp, dim=0)
        for prev in used:
            cn = cn - (cn @ prev) * prev
        nn = cn.norm()
        if nn < 1e-8:
            out["unique_ve"][name] = 0.0
            continue
        cn = cn / nn
        before = (residual.norm() ** 2).item()
        residual = residual - (residual @ cn) * cn
        after = (residual.norm() ** 2).item()
        out["unique_ve"][name] = (before - after) / total if total > 0 else 0.0
        used.append(cn)
    out["residual_ve"] = (residual.norm() ** 2).item() / total if total > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Full-completion teacher-forced log-prob signal
#
# For each (prompt, completion) pair we compute the model's mean per-token
# log-probability of the completion under teacher forcing. This replaces an
# earlier first-token-only signal: the first-token signal is dominated by
# opening-word style differences (e.g. "I" vs "The") between sycophantic and
# therapeutic completions and does not reflect the full continuation the
# model would produce. Summing over the truncated completion is more
# faithful to the model's actual completion preference.


@torch.no_grad()
def completion_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    n_completion_tokens: int = None,
) -> Tuple[float, int]:
    """Mean per-token log-prob of `completion` given `prompt` under teacher
    forcing. Returns (mean_logprob, n_tokens_scored)."""
    formatted = format_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(formatted, return_tensors="pt")
    full_ids = tokenizer.encode(formatted + completion, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]
    assert prompt_len <= full_ids.shape[1] and (
        full_ids[0, :prompt_len].tolist() == prompt_ids[0].tolist()
    ), "tokenizer is retokenizing across the prompt/completion boundary"
    if n_completion_tokens is not None:
        full_ids = full_ids[:, : prompt_len + n_completion_tokens]
    full_ids = full_ids.to(get_device(model))
    n_score = full_ids.shape[1] - prompt_len
    if n_score <= 0:
        return 0.0, 0
    logits = model(full_ids).logits  # (1, seq, vocab)
    # Predictions for token i live at logits[i-1]. We need logits at positions
    # [prompt_len-1, ..., prompt_len + n_score - 2] to predict completion
    # tokens at positions [prompt_len, ..., prompt_len + n_score - 1].
    pred_logits = logits[0, prompt_len - 1 : prompt_len - 1 + n_score, :].float()
    target_ids = full_ids[0, prompt_len : prompt_len + n_score]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_lps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(token_lps.mean().item()), int(n_score)


@torch.no_grad()
def completion_logprob_with_hook(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    layer: int,
    hook_fn,
    n_completion_tokens: int = None,
) -> Tuple[float, int]:
    """Same as completion_logprob, but with a forward hook attached to
    `model.model.layers[layer]` for the duration of the forward pass."""
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        return completion_logprob(model, tokenizer, prompt, completion, n_completion_tokens)
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Projection-ablation intervention


def projection_ablation_hook(direction: torch.Tensor):
    """Forward hook that subtracts the component of every position along `direction`.

    direction must be unit-norm. Operates on the residual stream output of a
    transformer block; preserves dtype and shape; safe under gradient-disabled
    inference.
    """

    def fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        d = direction.to(device=h.device, dtype=h.dtype)
        # h: (batch, seq, d_model). projection scalar per token.
        coef = (h @ d).unsqueeze(-1)  # (batch, seq, 1)
        h2 = h - coef * d
        if is_tuple:
            return (h2,) + out[1:]
        return h2

    return fn


def negative_steering_hook(direction: torch.Tensor, alpha: float):
    """Subtract alpha * direction from every residual-stream position."""

    def fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        d = direction.to(device=h.device, dtype=h.dtype)
        h2 = h - alpha * d
        if is_tuple:
            return (h2,) + out[1:]
        return h2

    return fn


def _completion_keys(pairs: List[Tuple[str, str]]) -> List[str]:
    """Distinct completion keys (e.g. 'sycophantic_completion') used across pairs."""
    keys = []
    for p, n in pairs:
        for k in (p, n):
            if k not in keys:
                keys.append(k)
    return keys


@torch.no_grad()
def compute_baseline_signals(
    model,
    tokenizer,
    stimuli: List[dict],
    pairs: List[Tuple[str, str]],
    n_completion_tokens: int = None,
) -> Dict[Tuple[str, str], List[float]]:
    """For each stimulus, one forward per distinct completion key, then build
    every (pos, neg) signal. Returns one list per pair."""
    keys = _completion_keys(pairs)
    out = {pair: [] for pair in pairs}
    for s in stimuli:
        per_key = {k: completion_logprob(model, tokenizer, s["user_prompt"], s[k],
                                         n_completion_tokens)[0] for k in keys}
        for pos_key, neg_key in pairs:
            out[(pos_key, neg_key)].append(per_key[pos_key] - per_key[neg_key])
    return out


@torch.no_grad()
def compute_intervention_signals(
    model,
    tokenizer,
    stimuli: List[dict],
    layer: int,
    direction: torch.Tensor,
    pairs: List[Tuple[str, str]],
    intervention: str = "ablation",
    alpha: float = 1.0,
    n_completion_tokens: int = None,
) -> Dict[Tuple[str, str], List[float]]:
    """Same as compute_baseline_signals but each forward has the intervention
    hook attached at `layer` for the entire prompt+completion forward pass."""
    if intervention == "ablation":
        hook_fn = projection_ablation_hook(direction)
    elif intervention == "negative_steering":
        hook_fn = negative_steering_hook(direction, alpha)
    else:
        raise ValueError(f"unknown intervention: {intervention}")

    keys = _completion_keys(pairs)
    out = {pair: [] for pair in pairs}
    for s in stimuli:
        per_key = {k: completion_logprob_with_hook(model, tokenizer, s["user_prompt"],
                                                   s[k], layer, hook_fn,
                                                   n_completion_tokens)[0] for k in keys}
        for pos_key, neg_key in pairs:
            out[(pos_key, neg_key)].append(per_key[pos_key] - per_key[neg_key])
    return out


def shift_summary(baseline: List[float], intervened: List[float]) -> Dict[str, float]:
    base = np.asarray(baseline)
    abl = np.asarray(intervened)
    diffs = abl - base
    return {
        "baseline_mean": float(base.mean()),
        "intervened_mean": float(abl.mean()),
        "shift_mean": float(diffs.mean()),
        "shift_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "shift_se": float(diffs.std(ddof=1) / np.sqrt(max(len(diffs), 1))) if len(diffs) > 1 else 0.0,
        "n": int(len(diffs)),
    }


# ---------------------------------------------------------------------------
# Main pipeline


def pick_layers(n_layers: int, n_keep: int = None) -> List[int]:
    """Sampled layer indices.

    Default returns every layer (n_keep=None). If n_keep is set, samples that
    many roughly-evenly-spaced layers including the first and last; this is
    used by the Colab path for the 32-layer 7B model where the full sweep is
    expensive but not impossible.
    """
    if n_keep is None or n_layers <= n_keep:
        return list(range(n_layers))
    step = max(1, n_layers // (n_keep - 1))
    layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    return layers


def stratified_sample(stimuli: List[dict], n_per_subcat: int = None) -> List[dict]:
    """Take up to n_per_subcat items per subcategory, deterministically.

    n_per_subcat=None returns all items.
    """
    by_cat: Dict[str, List[dict]] = {}
    for s in stimuli:
        by_cat.setdefault(s["subcategory"], []).append(s)
    out = []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda x: x["id"])
        if n_per_subcat is not None:
            items = items[:n_per_subcat]
        out.extend(items)
    return out


def run(args) -> None:
    set_seeds()
    device = select_device(args.device)
    print(f"Device: {device}")

    model_id = MODEL_IDS[args.model]
    print(f"Loading {model_id} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # bfloat16 on cuda (more stable than fp16 for OLMo-3 7B); fp32 elsewhere.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s; n_layers={model.config.num_hidden_layers}, "
          f"d_model={model.config.hidden_size}")

    n_layers = model.config.num_hidden_layers
    layers = pick_layers(n_layers, n_keep=args.n_layers if args.n_layers > 0 else None)
    target_layer = layers[len(layers) // 2]
    print(f"Sampling {len(layers)} layers: {layers}; pre-registered target layer = L{target_layer}")

    # ------------------------------------------------------------------
    # Stimuli — use the full distortion set and a matched factual set.
    raw_dist = load_json(STIM_DIR / "cognitive_distortions.json")
    raw_fact = load_json(STIM_DIR / "v2_factual_control.json")

    dist_stim = stratified_sample(raw_dist, args.n_per_cat if args.n_per_cat > 0 else None)
    fact_stim = sorted(raw_fact, key=lambda x: x["id"])[: len(dist_stim)]
    n_subcats = len(set(s['subcategory'] for s in dist_stim))
    per_cat_str = f"{args.n_per_cat}/subcat" if args.n_per_cat > 0 else "all"
    print(f"Distortion stimuli: {len(dist_stim)} ({per_cat_str} across {n_subcats} subcats)")
    print(f"Factual control stimuli: {len(fact_stim)}")

    # ------------------------------------------------------------------
    # Activation extraction
    syc_acts, ther_acts = extract_paired(
        model, tokenizer, dist_stim, "sycophantic_completion",
        "therapeutic_completion", layers, desc="syc/ther",
    )
    ther_acts2, cold_acts = extract_paired(
        model, tokenizer, dist_stim, "therapeutic_completion",
        "cold_completion", layers, desc="ther/cold",
    )
    fact_syc, fact_ther = extract_paired(
        model, tokenizer, fact_stim, "sycophantic_completion",
        "therapeutic_completion", layers, desc="factual",
    )

    # ------------------------------------------------------------------
    # E1: distortion-acceptance direction & probe
    print("\n[E1] Distortion-acceptance direction")
    d_dist = contrastive_direction(syc_acts, ther_acts)
    e1_within = within_domain_probe(syc_acts, ther_acts, layers)
    e1_factual_within = within_domain_probe(fact_syc, fact_ther, layers)
    # Permutation test for E1: shuffle the syc/ther labels and check whether
    # the within-domain probe still achieves the observed AUC. This tests
    # whether the recovered direction is information about the contrast or
    # a finite-sample artifact.
    rng = np.random.RandomState(SEED)
    null_aucs = []
    pool = syc_acts + ther_acts
    n_pos = len(syc_acts)
    for _ in tqdm(range(args.n_perms), desc="E1 perm test"):
        perm = rng.permutation(len(pool))
        pa = [pool[i] for i in perm[:n_pos]]
        pb = [pool[i] for i in perm[n_pos:]]
        sub = within_domain_probe(pa, pb, [target_layer], cv=5)
        null_aucs.append(sub[target_layer]["auc_mean"])
    obs_auc = e1_within[target_layer]["auc_mean"]
    null_aucs = np.asarray([a for a in null_aucs if not np.isnan(a)])
    p_perm = float(np.mean(null_aucs >= obs_auc)) if len(null_aucs) > 0 else float("nan")
    print(f"  L{target_layer} within AUC = {obs_auc:.3f}; null AUC mean = {null_aucs.mean():.3f}; p = {p_perm:.3f}")

    # ------------------------------------------------------------------
    # E2: empathy disentanglement
    print("\n[E2] Empathy disentanglement")
    d_warmth = contrastive_direction(ther_acts2, cold_acts)
    d_factual = contrastive_direction(fact_syc, fact_ther)
    cos_dist_warmth = cosine_by_layer(d_dist, d_warmth)
    cos_dist_factual = cosine_by_layer(d_dist, d_factual)
    decomps = {}
    decomps_factual_first = {}
    for l in layers:
        target = d_dist[l]
        # warmth-first order
        decomps[l] = decompose_direction(target, {"warmth": d_warmth[l], "factual": d_factual[l]})
        # factual-first order (Gram-Schmidt is order-sensitive; we report both)
        decomps_factual_first[l] = decompose_direction(
            target, {"factual": d_factual[l], "warmth": d_warmth[l]})
    print(f"  L{target_layer} cos(dist, warmth) = {cos_dist_warmth[target_layer]:.3f}")
    print(f"  L{target_layer} cos(dist, factual) = {cos_dist_factual[target_layer]:.3f}")
    print(f"  L{target_layer} unique VE [warmth-first]: warmth={decomps[target_layer]['unique_ve']['warmth']:.3f}, "
          f"factual={decomps[target_layer]['unique_ve']['factual']:.3f}, "
          f"residual={decomps[target_layer]['residual_ve']:.3f}")
    print(f"  L{target_layer} unique VE [factual-first]: warmth={decomps_factual_first[target_layer]['unique_ve']['warmth']:.3f}, "
          f"factual={decomps_factual_first[target_layer]['unique_ve']['factual']:.3f}, "
          f"residual={decomps_factual_first[target_layer]['residual_ve']:.3f}")

    # ------------------------------------------------------------------
    # E3: layer localization via projection magnitude
    print("\n[E3] Layer localization")
    proj_syc = project(syc_acts, d_dist)
    proj_ther = project(ther_acts, d_dist)
    sep_by_layer = {}
    for l in layers:
        a = np.array(proj_syc[l])
        b = np.array(proj_ther[l])
        # standardized mean difference (Cohen's d)
        sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                     max(len(a) + len(b) - 2, 1))
        d = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
        sep_by_layer[l] = {
            "syc_mean": float(a.mean()), "ther_mean": float(b.mean()),
            "cohens_d": float(d), "auc": float(e1_within[l]["auc_mean"]),
        }
    # Best descriptive layer = highest Cohen's d in the middle 80% of layers
    # (excludes the embedding-adjacent and unembedding-adjacent layers, which
    # often achieve trivially high probe AUC for surface-form reasons).
    interior = [l for l in layers if 1 <= layers.index(l) < len(layers) - 1]
    best_layer = max(interior, key=lambda l: sep_by_layer[l]["cohens_d"]) if interior else layers[len(layers) // 2]
    print(f"  Best descriptive layer (interior, by Cohen's d) = L{best_layer} "
          f"(AUC={sep_by_layer[best_layer]['auc']:.3f}, Cohen's d={sep_by_layer[best_layer]['cohens_d']:.2f})")
    # Pre-registered intervention layer (median sampled layer) - decided
    # before E5/E6 results are seen. Distinct from descriptive best layer
    # to avoid p-hacking the intervention success.
    intervention_layer = target_layer
    print(f"  Pre-registered intervention layer = L{intervention_layer}")

    # ------------------------------------------------------------------
    # E4: cross-distortion LOO generalization at every sampled layer
    print("\n[E4] Cross-distortion leave-one-out (sweep over all sampled layers)")
    by_cat: Dict[str, List[int]] = {}
    for i, s in enumerate(dist_stim):
        by_cat.setdefault(s["subcategory"], []).append(i)
    cats = sorted(by_cat)
    e4_per_layer = {l: {} for l in layers}
    for held in tqdm(cats, desc="E4 LOO"):
        train_idx = [i for c, ix in by_cat.items() if c != held for i in ix]
        test_idx = by_cat[held]
        if len(test_idx) < 2 or len(train_idx) < 4:
            continue
        sp = [syc_acts[i] for i in train_idx]
        sn = [ther_acts[i] for i in train_idx]
        tp = [syc_acts[i] for i in test_idx]
        tn = [ther_acts[i] for i in test_idx]
        res = cross_domain_probe(sp, sn, tp, tn, layers)
        for l in layers:
            e4_per_layer[l][held] = {
                "n_test": len(test_idx), "auc": res[l]["auc"], "acc": res[l]["acc"],
            }
    e4_layer_summary = {}
    for l in layers:
        aucs_l = [v["auc"] for v in e4_per_layer[l].values() if not np.isnan(v["auc"])]
        accs_l = [v["acc"] for v in e4_per_layer[l].values() if not np.isnan(v["acc"])]
        e4_layer_summary[l] = {
            "mean_auc": float(np.mean(aucs_l)) if aucs_l else float("nan"),
            "std_auc": float(np.std(aucs_l)) if aucs_l else float("nan"),
            "mean_acc": float(np.mean(accs_l)) if accs_l else float("nan"),
            "n_subcats": len(aucs_l),
        }
    e4_best_layer = max(e4_layer_summary, key=lambda l: e4_layer_summary[l]["mean_auc"]) \
        if any(not np.isnan(e4_layer_summary[l]["mean_auc"]) for l in layers) else layers[0]
    print(f"  LOO mean AUC by layer: " + ", ".join(
        f"L{l}={e4_layer_summary[l]['mean_auc']:.2f}" for l in layers))
    print(f"  Best LOO layer = L{e4_best_layer} "
          f"(mean AUC = {e4_layer_summary[e4_best_layer]['mean_auc']:.3f})")

    # ------------------------------------------------------------------
    # E7/E8/E9: per-subtype geometry of the distortion-validation subspace
    print("\n[E7/E8/E9] Per-subtype geometry across layers")
    subtype_dirs = per_subtype_directions(dist_stim, syc_acts, ther_acts, layers)
    geometry_by_layer = {}
    for l in layers:
        geometry_by_layer[l] = subtype_geometry(subtype_dirs, l, ks=(1, 2, 3, 5))
    g_target = geometry_by_layer[target_layer]
    n_cats = len(g_target["subcategories"])
    print(f"  L{target_layer} singular values (top 5): "
          + ", ".join(f"{s:.3f}" for s in g_target["singular_values"][:5]))
    print(f"  L{target_layer} cumulative variance (top 1,2,3,5): "
          f"{g_target['cumulative_var_fraction'][0]:.3f}, "
          f"{g_target['cumulative_var_fraction'][1]:.3f}, "
          f"{g_target['cumulative_var_fraction'][2]:.3f}, "
          f"{g_target['cumulative_var_fraction'][min(4, n_cats-1)]:.3f}")
    print(f"  L{target_layer} participation ratio = {g_target['participation_ratio']:.2f}  "
          f"(out of {n_cats} possible)")
    cm = np.array(g_target["pairwise_cosine"])
    off = cm[~np.eye(n_cats, dtype=bool)]
    print(f"  L{target_layer} pairwise cos: mean={off.mean():.3f}, "
          f"min={off.min():.3f}, max={off.max():.3f}")
    pv5 = g_target["subspace_variance_retained_top_k"].get(5, {})
    if pv5:
        worst = min(pv5, key=pv5.get); best = max(pv5, key=pv5.get)
        print(f"  L{target_layer} top-5-subspace variance retained: "
              f"min={worst}({pv5[worst]:.2f}), max={best}({pv5[best]:.2f})")

    # ------------------------------------------------------------------
    # E5/E6: exhaustive sweep of layer × intervention × alpha
    #
    # Pre-registered headline test:  intervention_layer (median sampled layer)
    #                                 negative_steering at alpha=4.0
    # We additionally sweep all sampled layers at alpha=4.0 for both
    # interventions, and a dose-response over alphas at intervention_layer.
    # All target effects are compared against random-direction controls
    # using the same sweep.
    print(f"\n[E5/E6] Causal intervention sweep")
    # Held-out intervention stimuli: the direction d_dist is fit on the full
    # dist_stim list (extraction above), so the intervention should be measured
    # on stimuli that are NOT in the fit set. We refit d_dist on the first half
    # and evaluate the intervention on a held-out slice from the second half.
    n_fit_half = len(dist_stim) // 2
    fit_stim = dist_stim[:n_fit_half]
    holdout_stim = dist_stim[n_fit_half:]
    n_intervene = (len(holdout_stim) if args.n_intervene <= 0
                   else min(args.n_intervene, len(holdout_stim)))
    inter_stim = holdout_stim[:n_intervene]
    # Refit d_dist on the fit half so the intervention is genuinely out-of-sample.
    syc_fit = [syc_acts[i] for i in range(n_fit_half)]
    ther_fit = [ther_acts[i] for i in range(n_fit_half)]
    print(f"  Held-out split: fit on {n_fit_half} stimuli, intervene on "
          f"{len(inter_stim)} held-out stimuli (out-of-sample direction).")
    pairs = [("therapeutic_completion", "sycophantic_completion"),  # E5
             ("therapeutic_completion", "cold_completion")]         # E6
    pair_keys = {pairs[0]: "E5_ther_vs_syc", pairs[1]: "E6_ther_vs_cold"}

    # Single baseline computation (one fwd per stim).
    print(f"  Computing baseline signals (n={n_intervene}) ...")
    baseline_signals = compute_baseline_signals(model, tokenizer, inter_stim, pairs)

    # Pre-generate random vectors used everywhere for fair comparison.
    rng = np.random.RandomState(SEED)
    d_model = d_dist[intervention_layer].shape[0]
    random_dirs = []
    for _ in range(args.n_random):
        rv = torch.from_numpy(rng.randn(d_model).astype(np.float32))
        random_dirs.append(F.normalize(rv, dim=0))

    def run_intervention(layer: int, direction: torch.Tensor, intervention: str, alpha: float):
        sigs = compute_intervention_signals(
            model, tokenizer, inter_stim, layer, direction, pairs,
            intervention=intervention, alpha=alpha,
        )
        return {
            pair_keys[p]: shift_summary(baseline_signals[p], sigs[p])
            for p in pairs
        }

    def random_controls(layer: int, intervention: str, alpha: float):
        runs = []
        for rv in random_dirs:
            sigs = compute_intervention_signals(
                model, tokenizer, inter_stim, layer, rv, pairs,
                intervention=intervention, alpha=alpha,
            )
            runs.append({pair_keys[p]: shift_summary(baseline_signals[p], sigs[p])
                         for p in pairs})
        return runs

    # Layer sweep at alpha=args.alpha for both interventions.
    # For each sampled layer we refit d_dist on the fit half of the stimuli so
    # the direction at every layer is out-of-sample for the held-out intervention.
    d_dist_layer_heldout = contrastive_direction(syc_fit, ther_fit)
    print(f"  Sweeping {len(layers)} layers x 2 interventions at alpha={args.alpha} "
          f"(held-out direction fit on {len(fit_stim)} / {len(dist_stim)}) ...")
    sweep = {}  # sweep[layer][intervention] = {target: {...}, random: [...]}
    for li, l in enumerate(tqdm(layers, desc="E5/E6 layer sweep")):
        sweep[l] = {}
        direction = d_dist_layer_heldout[l]
        for intervention, alpha in [("ablation", 0.0), ("negative_steering", args.alpha)]:
            target = run_intervention(l, direction, intervention, alpha)
            rand = random_controls(l, intervention, alpha)
            rand_shifts = {pair_keys[p]: [r[pair_keys[p]]["shift_mean"] for r in rand] for p in pairs}
            zscores = {}
            for p in pairs:
                key = pair_keys[p]
                rs = np.asarray(rand_shifts[key])
                zscores[key] = (
                    (target[key]["shift_mean"] - rs.mean()) / max(rs.std(ddof=1), 1e-8)
                    if rs.size > 1 else float("nan")
                )
            # Ablation has no alpha; record None to avoid misleading downstream
            # consumers into thinking the projection-ablation was scaled.
            recorded_alpha = None if intervention == "ablation" else alpha
            sweep[l][intervention] = {"alpha": recorded_alpha, "target": target,
                                      "random_shifts": rand_shifts, "z_vs_random": zscores}

    # Identify (layer, intervention) with maximum specificity = |E5| / |E6|
    # among configurations where E5 shift is in the expected direction
    # (positive shift = increases ther over syc preference).
    #
    # Two "best" configs are tracked:
    #   * unrestricted: argmax of |E5|/|E6| (can be inflated when |E6| is
    #     a few-stimulus-noise floor near zero)
    #   * E6_meaningful: argmax restricted to configs where the E5 shift is
    #     at least 50% of the pre-registered headline E5, so the ratio is
    #     not dominated by a tiny target effect.
    head_e5_abs = abs(sweep[intervention_layer]["negative_steering"]
                      ["target"]["E5_ther_vs_syc"]["shift_mean"])
    e5_floor = 0.5 * head_e5_abs
    best_cfg = None
    best_spec = -1.0
    best_cfg_meaningful = None
    best_spec_meaningful = -1.0
    for l in layers:
        for intervention in ("ablation", "negative_steering"):
            t = sweep[l][intervention]["target"]
            e5s = t["E5_ther_vs_syc"]["shift_mean"]
            e6s = t["E6_ther_vs_cold"]["shift_mean"]
            if e5s <= 0:
                continue
            spec = e5s / max(abs(e6s), 1e-8)
            if spec > best_spec:
                best_spec = spec
                best_cfg = (l, intervention, e5s, e6s, spec)
            if e5s >= e5_floor and spec > best_spec_meaningful:
                best_spec_meaningful = spec
                best_cfg_meaningful = (l, intervention, e5s, e6s, spec)
    if best_cfg:
        l, intervention, e5s, e6s, spec = best_cfg
        denom_caveat = " (DENOMINATOR-NEAR-ZERO; |E6| < 0.01)" if abs(e6s) < 0.01 else ""
        print(f"  Best specificity (positive E5): L{l} {intervention}, "
              f"E5={e5s:+.3f}, E6={e6s:+.3f}, specificity={spec:.2f}{denom_caveat}")
    if best_cfg_meaningful:
        l, intervention, e5s, e6s, spec = best_cfg_meaningful
        print(f"  Best specificity (E5 >= 50% of headline): L{l} {intervention}, "
              f"E5={e5s:+.3f}, E6={e6s:+.3f}, specificity={spec:.2f}")

    # Alpha dose-response at intervention_layer for negative_steering
    # (uses the same held-out direction as the layer sweep).
    alpha_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    print(f"  Alpha dose-response at L{intervention_layer} (negative_steering): "
          f"alphas={alpha_grid} ...")
    alpha_sweep = {}
    direction = d_dist_layer_heldout[intervention_layer]
    for a in tqdm(alpha_grid, desc="alpha sweep"):
        target = run_intervention(intervention_layer, direction, "negative_steering", a)
        rand = random_controls(intervention_layer, "negative_steering", a)
        rand_shifts = {pair_keys[p]: [r[pair_keys[p]]["shift_mean"] for r in rand] for p in pairs}
        zscores = {}
        for p in pairs:
            key = pair_keys[p]
            rs = np.asarray(rand_shifts[key])
            zscores[key] = (
                (target[key]["shift_mean"] - rs.mean()) / max(rs.std(ddof=1), 1e-8)
                if rs.size > 1 else float("nan")
            )
        alpha_sweep[a] = {"target": target, "random_shifts": rand_shifts, "z_vs_random": zscores}

    # Pre-registered headline result + a higher-n random null at the headline
    # configuration only (cheap because it's one (layer, intervention) point).
    pre_reg = sweep[intervention_layer]["negative_steering"]
    n_random_headline = max(args.n_random_headline, args.n_random)
    print(f"  Headline higher-n random null: n_random_headline={n_random_headline} ...")
    rng2 = np.random.RandomState(SEED + 1)
    rand_dirs_h = [F.normalize(torch.from_numpy(rng2.randn(d_model).astype(np.float32)), dim=0)
                   for _ in range(n_random_headline)]
    rand_h = []
    for rv in tqdm(rand_dirs_h, desc="headline null"):
        sigs = compute_intervention_signals(
            model, tokenizer, inter_stim, intervention_layer, rv, pairs,
            intervention="negative_steering", alpha=args.alpha,
        )
        rand_h.append({pair_keys[p]: shift_summary(baseline_signals[p], sigs[p])
                      for p in pairs})
    rand_h_shifts = {pair_keys[p]: np.asarray(
        [r[pair_keys[p]]["shift_mean"] for r in rand_h]) for p in pairs}
    headline_z = {pair_keys[p]: (
        (pre_reg["target"][pair_keys[p]]["shift_mean"] - rand_h_shifts[pair_keys[p]].mean())
        / max(rand_h_shifts[pair_keys[p]].std(ddof=1), 1e-8)
    ) for p in pairs}
    headline = {
        "layer": intervention_layer,
        "intervention": "negative_steering",
        "alpha": args.alpha,
        "E5_shift_mean": pre_reg["target"]["E5_ther_vs_syc"]["shift_mean"],
        "E5_shift_se": pre_reg["target"]["E5_ther_vs_syc"]["shift_se"],
        "E5_z_vs_random_n5": pre_reg["z_vs_random"]["E5_ther_vs_syc"],
        "E5_z_vs_random_headline": float(headline_z["E5_ther_vs_syc"]),
        "E5_z_vs_random": float(headline_z["E5_ther_vs_syc"]),
        "E6_shift_mean": pre_reg["target"]["E6_ther_vs_cold"]["shift_mean"],
        "E6_shift_se": pre_reg["target"]["E6_ther_vs_cold"]["shift_se"],
        "E6_z_vs_random_n5": pre_reg["z_vs_random"]["E6_ther_vs_cold"],
        "E6_z_vs_random_headline": float(headline_z["E6_ther_vs_cold"]),
        "E6_z_vs_random": float(headline_z["E6_ther_vs_cold"]),
        "n_random_headline": n_random_headline,
        "headline_random_shift_mean_E5": float(rand_h_shifts["E5_ther_vs_syc"].mean()),
        "headline_random_shift_std_E5": float(rand_h_shifts["E5_ther_vs_syc"].std(ddof=1)),
        "headline_random_shift_mean_E6": float(rand_h_shifts["E6_ther_vs_cold"].mean()),
        "headline_random_shift_std_E6": float(rand_h_shifts["E6_ther_vs_cold"].std(ddof=1)),
    }
    spec_pre = abs(headline["E5_shift_mean"]) / max(abs(headline["E6_shift_mean"]), 1e-8)
    headline["specificity_ratio"] = spec_pre

    # Paired-bootstrap 95% CI on the specificity ratio at the headline
    # configuration. We recompute the target intervention signals once on the
    # held-out stimuli, derive per-stimulus shifts, and resample with
    # replacement to get the distribution of |mean E5| / |mean E6|.
    sigs_for_ci = compute_intervention_signals(
        model, tokenizer, inter_stim, intervention_layer, direction, pairs,
        intervention="negative_steering", alpha=args.alpha,
    )
    per_stim_E5 = np.asarray(sigs_for_ci[pairs[0]]) - np.asarray(baseline_signals[pairs[0]])
    per_stim_E6 = np.asarray(sigs_for_ci[pairs[1]]) - np.asarray(baseline_signals[pairs[1]])
    rng_boot = np.random.RandomState(SEED + 100)
    n_boot = 5000
    n_stim = len(per_stim_E5)
    boot_ratios = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng_boot.choice(n_stim, n_stim, replace=True)
        mE5 = per_stim_E5[idx].mean()
        mE6 = per_stim_E6[idx].mean()
        boot_ratios[i] = abs(mE5) / max(abs(mE6), 1e-8)
    headline["specificity_ratio_bootstrap_95_ci"] = [
        float(np.percentile(boot_ratios, 2.5)),
        float(np.percentile(boot_ratios, 97.5)),
    ]
    headline["specificity_ratio_bootstrap_mean"] = float(boot_ratios.mean())
    headline["specificity_ratio_prob_above_1"] = float((boot_ratios > 1.0).mean())
    headline["specificity_ratio_n_boot"] = n_boot

    print(f"  Pre-registered headline (L{intervention_layer}, negative_steering, alpha={args.alpha}):")
    print(f"    E5 shift = {headline['E5_shift_mean']:+.3f} "
          f"(SE={headline['E5_shift_se']:.3f}, z={headline['E5_z_vs_random']:+.2f})")
    print(f"    E6 shift = {headline['E6_shift_mean']:+.3f} "
          f"(SE={headline['E6_shift_se']:.3f}, z={headline['E6_z_vs_random']:+.2f})")
    print(f"    specificity |E5|/|E6| = {spec_pre:.2f}")

    # ------------------------------------------------------------------
    # Save results
    out = {
        "config": {
            "model_id": model_id,
            "n_layers": n_layers,
            "sampled_layers": layers,
            "target_layer_pre_registered": target_layer,
            "intervention_layer_pre_registered": intervention_layer,
            "best_descriptive_layer": best_layer,
            "best_loo_layer": e4_best_layer,
            "n_distortion_stim": len(dist_stim),
            "n_factual_stim": len(fact_stim),
            "n_intervene": n_intervene,
            "n_random_controls": args.n_random,
            "n_completion_tokens": "full",
            "alpha_layer_sweep": args.alpha,
            "seed": SEED,
        },
        "E1_distortion_direction": {
            "within_domain_probe": e1_within,
            "factual_within_domain_probe": e1_factual_within,
            "permutation_target_layer": {
                "layer": target_layer,
                "observed_auc": obs_auc,
                "null_mean_auc": float(null_aucs.mean()),
                "null_std_auc": float(null_aucs.std()),
                "p_value": p_perm,
                "n_perms": int(len(null_aucs)),
            },
        },
        "E2_disentanglement": {
            "cos_dist_warmth_by_layer": cos_dist_warmth,
            "cos_dist_factual_by_layer": cos_dist_factual,
            "decomp_by_layer_warmth_first": decomps,
            "decomp_by_layer_factual_first": decomps_factual_first,
            # Backward-compatible alias for figure-builder code that expects
            # "decomp_by_layer": defaults to the warmth-first ordering.
            "decomp_by_layer": decomps,
        },
        "E3_layer_localization": {"by_layer": sep_by_layer, "best_layer": best_layer},
        "E4_cross_distortion_loo": {
            "by_layer": {l: e4_layer_summary[l] for l in layers},
            "by_layer_per_subcat": {l: e4_per_layer[l] for l in layers},
            "best_layer": e4_best_layer,
        },
        "E7_E8_E9_geometry": {
            "by_layer": geometry_by_layer,
        },
        "E5_E6_intervention_sweep": {
            "layer_sweep": sweep,
            "alpha_sweep_at_intervention_layer": alpha_sweep,
            "headline_pre_registered": headline,
            "best_specificity_config": {
                "layer": best_cfg[0] if best_cfg else None,
                "intervention": best_cfg[1] if best_cfg else None,
                "E5_shift": best_cfg[2] if best_cfg else None,
                "E6_shift": best_cfg[3] if best_cfg else None,
                "specificity": best_cfg[4] if best_cfg else None,
                "denominator_near_zero": (abs(best_cfg[3]) < 0.01) if best_cfg else None,
            },
            "best_specificity_config_meaningful": {
                "criterion": "argmax(|E5|/|E6|) restricted to E5 >= 50% of pre-registered headline E5",
                "headline_E5_50pct_floor": e5_floor,
                "layer": best_cfg_meaningful[0] if best_cfg_meaningful else None,
                "intervention": best_cfg_meaningful[1] if best_cfg_meaningful else None,
                "E5_shift": best_cfg_meaningful[2] if best_cfg_meaningful else None,
                "E6_shift": best_cfg_meaningful[3] if best_cfg_meaningful else None,
                "specificity": best_cfg_meaningful[4] if best_cfg_meaningful else None,
            },
        },
    }
    save_json(out, RESULTS_DIR / "results.json")
    print(f"\nSaved results to {RESULTS_DIR / 'results.json'}")

    # ------------------------------------------------------------------
    # Figures
    make_figures(out, layers, target_layer, best_layer, intervention_layer,
                 sweep, alpha_sweep, alpha_grid, args.alpha,
                 geometry_by_layer)
    print(f"Saved figures to {FIGURES_DIR}")


def _at(d, key):
    return d[str(key)] if str(key) in d else d[key]


def make_figures(res: dict, layers: List[int], target_layer: int, best_layer: int,
                 intervention_layer: int, sweep: dict, alpha_sweep: dict,
                 alpha_grid: List[float], default_alpha: float,
                 geometry_by_layer: dict = None) -> None:
    # Figure 1: per-layer AUC for distortion vs factual probes
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    e1 = res["E1_distortion_direction"]["within_domain_probe"]
    fac = res["E1_distortion_direction"]["factual_within_domain_probe"]
    auc_d = [_at(e1, l)["auc_mean"] for l in layers]
    auc_f = [_at(fac, l)["auc_mean"] for l in layers]
    ax.plot(layers, auc_d, "o-", color=COLOR["dist"], label="distortion (syc vs ther)")
    ax.plot(layers, auc_f, "s--", color=COLOR["factual"], label="factual (syc vs ther)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvline(best_layer, color=COLOR["neutral"], lw=0.8, ls=":",
               label=f"interior best L{best_layer}")
    ax.set_xlabel("layer")
    ax.set_ylabel("within-domain probe AUC (5-fold CV)")
    ax.set_title("E1/E3: Distortion-acceptance is recoverable per-layer")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_layer_auc.png")
    plt.close(fig)

    # Figure 2: cosine sim with warmth and factual
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    e2 = res["E2_disentanglement"]
    cw_v = [_at(e2["cos_dist_warmth_by_layer"], l) for l in layers]
    cf_v = [_at(e2["cos_dist_factual_by_layer"], l) for l in layers]
    ax.plot(layers, cw_v, "o-", color=COLOR["warmth"], label="cos(dist, warmth)")
    ax.plot(layers, cf_v, "s-", color=COLOR["factual"], label="cos(dist, factual)")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity")
    ax.set_title("E2: Distortion-acceptance vs warmth and factual directions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_cosine.png")
    plt.close(fig)

    # Figure 3: variance decomposition
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    decomps = e2["decomp_by_layer"]
    warmth_ve = [_at(decomps, l)["unique_ve"]["warmth"] for l in layers]
    factual_ve = [_at(decomps, l)["unique_ve"]["factual"] for l in layers]
    residual_ve = [_at(decomps, l)["residual_ve"] for l in layers]
    ax.stackplot(layers, warmth_ve, factual_ve, residual_ve,
                 labels=["warmth (unique)", "factual (unique)", "residual"],
                 colors=[COLOR["warmth"], COLOR["factual"], COLOR["neutral"]], alpha=0.85)
    ax.set_xlabel("layer")
    ax.set_ylabel("variance fraction of distortion direction")
    ax.set_title("E2: Gram-Schmidt variance decomposition")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_decomposition.png")
    plt.close(fig)

    # Figure 4a: LOO AUC by layer
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    e4 = res["E4_cross_distortion_loo"]
    by_layer = e4["by_layer"]
    means = [_at(by_layer, l)["mean_auc"] for l in layers]
    stds = [_at(by_layer, l)["std_auc"] for l in layers]
    ax.errorbar(layers, means, yerr=stds, fmt="o-", color=COLOR["dist"],
                capsize=3, label="LOO mean ± std")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    best_loo = e4["best_layer"]
    ax.axvline(best_loo, color=COLOR["neutral"], lw=0.8, ls=":",
               label=f"best LOO L{best_loo}")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("E4: Cross-distortion LOO generalization across layers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4a_loo_by_layer.png")
    plt.close(fig)

    # Figure 4b: LOO AUC by held-out subcategory at best LOO layer
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    per_subcat = _at(e4["by_layer_per_subcat"], best_loo)
    cats = sorted(per_subcat.keys())
    aucs = [per_subcat[c]["auc"] for c in cats]
    ax.bar(range(len(cats)), aucs, color=COLOR["dist"], alpha=0.85)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axhline(float(np.nanmean(aucs)), color="k", lw=1.0, ls="--", label="mean AUC")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("held-out AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"E4: LOO by held-out distortion type (L{best_loo})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4b_loo_by_subcat.png")
    plt.close(fig)

    # Figure 5: intervention layer sweep (negative_steering)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    for ax, intervention in zip(axes, ("ablation", "negative_steering")):
        e5_means = [sweep[l][intervention]["target"]["E5_ther_vs_syc"]["shift_mean"] for l in layers]
        e5_ses = [sweep[l][intervention]["target"]["E5_ther_vs_syc"]["shift_se"] for l in layers]
        e6_means = [sweep[l][intervention]["target"]["E6_ther_vs_cold"]["shift_mean"] for l in layers]
        e6_ses = [sweep[l][intervention]["target"]["E6_ther_vs_cold"]["shift_se"] for l in layers]
        rand_e5 = [np.mean(sweep[l][intervention]["random_shifts"]["E5_ther_vs_syc"]) for l in layers]
        rand_e6 = [np.mean(sweep[l][intervention]["random_shifts"]["E6_ther_vs_cold"]) for l in layers]
        ax.errorbar(layers, e5_means, yerr=e5_ses, fmt="o-", color=COLOR["dist"],
                    label="E5 (ther vs syc)", capsize=3)
        ax.errorbar(layers, e6_means, yerr=e6_ses, fmt="s-", color=COLOR["warmth"],
                    label="E6 (ther vs cold)", capsize=3)
        ax.plot(layers, rand_e5, ":", color=COLOR["dist"], alpha=0.6, label="E5 random")
        ax.plot(layers, rand_e6, ":", color=COLOR["warmth"], alpha=0.6, label="E6 random")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("layer")
        title = intervention if intervention == "ablation" else f"negative steering (alpha={default_alpha})"
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel(r"$\Delta$ log-prob signal  (intervention - baseline)")
    fig.suptitle("E5/E6: Intervention shift across all sampled layers", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_layer_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 6: alpha dose-response at intervention layer (negative_steering)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    e5_means = [alpha_sweep[a]["target"]["E5_ther_vs_syc"]["shift_mean"] for a in alpha_grid]
    e5_ses = [alpha_sweep[a]["target"]["E5_ther_vs_syc"]["shift_se"] for a in alpha_grid]
    e6_means = [alpha_sweep[a]["target"]["E6_ther_vs_cold"]["shift_mean"] for a in alpha_grid]
    e6_ses = [alpha_sweep[a]["target"]["E6_ther_vs_cold"]["shift_se"] for a in alpha_grid]
    ax.errorbar(alpha_grid, e5_means, yerr=e5_ses, fmt="o-", color=COLOR["dist"],
                label="E5 (ther vs syc)", capsize=3)
    ax.errorbar(alpha_grid, e6_means, yerr=e6_ses, fmt="s-", color=COLOR["warmth"],
                label="E6 (ther vs cold)", capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"steering magnitude $\alpha$")
    ax.set_ylabel(r"$\Delta$ log-prob signal")
    ax.set_xscale("log")
    ax.set_title(f"E5/E6: Dose response (negative steering at L{intervention_layer})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_alpha_sweep.png")
    plt.close(fig)

    if geometry_by_layer is None:
        return

    # Figure 7: scree plot — singular values and cumulative variance per layer
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    g_target = geometry_by_layer[target_layer]
    n_cat = len(g_target["singular_values"])
    cmap = plt.get_cmap("viridis")
    for i, l in enumerate(layers):
        g = geometry_by_layer[l]
        c = cmap(i / max(len(layers) - 1, 1))
        axes[0].plot(range(1, n_cat + 1), g["singular_values"], "o-", color=c,
                     alpha=0.85, label=f"L{l}", lw=1.0, ms=3)
        axes[1].plot(range(1, n_cat + 1), g["cumulative_var_fraction"], "o-", color=c,
                     alpha=0.85, label=f"L{l}", lw=1.0, ms=3)
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("singular value")
    axes[0].set_title("E8: SVD singular values across layers")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[1].set_xlabel("number of principal components")
    axes[1].set_ylabel("cumulative variance fraction")
    axes[1].set_title("E8: Cumulative variance of subtype subspace")
    axes[1].set_ylim(0, 1.02)
    axes[1].axhline(0.9, color="k", lw=0.5, ls=":")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_geometry_svd.png")
    plt.close(fig)

    # Figure 8: pairwise cosine heatmap at the pre-registered target layer
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cm = np.array(g_target["pairwise_cosine"])
    cats = g_target["subcategories"]
    im = ax.imshow(cm, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cats))); ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(cats, fontsize=7)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(cm[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"E9: Pairwise cosine of per-subtype directions (L{target_layer})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_geometry_cosine.png")
    plt.close(fig)

    # Figure 9: participation ratio across layers + variance retained per subtype
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    pr = [geometry_by_layer[l]["participation_ratio"] for l in layers]
    axes[0].plot(layers, pr, "o-", color=COLOR["purple"])
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("participation ratio")
    axes[0].set_title("E8: Effective dimensionality of subtype subspace")
    axes[0].axhline(1, color="k", lw=0.5, ls=":")
    axes[0].axhline(n_cat, color="k", lw=0.5, ls=":")
    axes[0].set_ylim(0.8, n_cat + 0.5)

    # variance retained per subtype, top-3 subspace, at target layer
    pv = g_target["subspace_variance_retained_top_k"].get(3, {})
    if pv:
        items = sorted(pv.items(), key=lambda kv: kv[1])
        axes[1].barh([k for k, _ in items], [v for _, v in items],
                     color=COLOR["dist"], alpha=0.85)
        axes[1].set_xlim(0, 1.02)
        axes[1].set_xlabel("variance fraction retained in top-3 subspace")
        axes[1].set_title(f"E9: How well each subtype lives in the consensus subspace (L{target_layer})")
        axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_geometry_subspace.png")
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), default="1b")
    ap.add_argument("--device", default=None,
                   help="cuda|mps|cpu; default: auto")
    ap.add_argument("--n-per-cat", type=int, default=0,
                   help="stimuli per distortion subcategory; 0 = all")
    ap.add_argument("--n-intervene", type=int, default=30,
                   help="stimuli for the intervention sweep; 0 = all distortion stimuli")
    ap.add_argument("--n-random", type=int, default=10,
                   help="random-direction controls per intervention configuration")
    ap.add_argument("--n-random-headline", type=int, default=30,
                   help="random-direction controls for the pre-registered headline (cheap; only at intervention_layer)")
    ap.add_argument("--n-perms", type=int, default=500,
                   help="permutations for the E1 within-domain probe null")
    ap.add_argument("--n-layers", type=int, default=0,
                   help="number of layers to sample; 0 = all layers")
    ap.add_argument("--alpha", type=float, default=4.0,
                   help="alpha for the layer sweep negative-steering condition")
    ap.add_argument("--quick", action="store_true",
                   help="smoke-test mode: tiny stimulus counts")
    args = ap.parse_args()
    if args.quick:
        args.n_per_cat = 2
        args.n_intervene = 6
        args.n_random = 2
        args.n_random_headline = 4
        args.n_perms = 30
        args.n_layers = 9
    return args


if __name__ == "__main__":
    run(parse_args())
