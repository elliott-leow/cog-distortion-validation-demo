"""Supplementary mitigation experiments for the limitations section.

Runs three additional, cheap analyses that are not part of the main
`reference.py` pipeline but directly address limitations flagged in
`paper/paper.md` §6:

  M1 (GroupKFold probe)          — Limitation 7 pairing caveat (§4.1).
                                   Re-run the within-domain probe with
                                   GroupKFold by user_prompt so that a
                                   stimulus's syc and ther activations
                                   never land on opposite sides of the
                                   train/test split. If AUC stays at 1.000
                                   the probe is not relying on prompt
                                   leakage; if AUC drops, the saturated
                                   StratifiedKFold result was partly
                                   leakage-driven.

  M2 (Per-subtype label-shuffle null for E7-E9)
                                 — Limitation 11 noise-floor caveat.
                                   For each subtype, shuffle the syc/ther
                                   labels within the subtype, recompute
                                   the per-subtype direction, stack into
                                   a (12 × d_model) matrix, and compute
                                   the participation ratio + mean
                                   pairwise cosine. Repeat n_shuffles=100
                                   times to form a null distribution
                                   against which the observed PR and
                                   cosine are compared.

  M3 (Held-out Cohen's d)       — Limitation 14 in-sample-bias caveat.
                                   Fit d_dist on the first half of the
                                   distortion stimuli and compute
                                   Cohen's d at each sampled layer on
                                   the held-out second half. Compare to
                                   the in-sample Cohen's d reported in
                                   §4.3 to quantify the optimism bias.

Saves `results/mitigations.json`. Independent of `reference.py`'s
results.json; `fill_paper.py` references this file separately.

Usage:
    python mitigation_experiments.py --device mps
    python mitigation_experiments.py --model 7b --device cuda  # Colab
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-use helpers from reference.py
import reference  # noqa: E402

SEED = 42
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def set_seeds(s: int = SEED) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def group_kfold_probe(pos_acts, neg_acts, groups_pos, groups_neg,
                      layers: List[int], n_splits: int = 5) -> Dict[int, dict]:
    """Within-domain logistic probe with GroupKFold by user_prompt id so
    that each stimulus's syc and ther activations are kept together.

    Returns {layer: {auc_mean, auc_std, acc_mean, acc_std}}.
    """
    out = {}
    for l in layers:
        X = np.concatenate([
            np.stack([a[l].numpy() for a in pos_acts]),
            np.stack([a[l].numpy() for a in neg_acts]),
        ])
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
        groups = np.concatenate([groups_pos, groups_neg])
        n_groups = len(np.unique(groups))
        n_cv = min(n_splits, n_groups)
        if n_cv < 2:
            out[l] = {"auc_mean": float("nan"), "auc_std": float("nan"),
                      "acc_mean": float("nan"), "acc_std": float("nan"),
                      "note": "insufficient groups"}
            continue
        gkf = GroupKFold(n_splits=n_cv)
        aucs, accs = [], []
        for tr, te in gkf.split(X, y, groups):
            clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            prob = clf.predict_proba(X[te])[:, 1]
            accs.append(accuracy_score(y[te], pred))
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except ValueError:
                aucs.append(float("nan"))
        out[l] = {
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "n_splits": n_cv,
        }
    return out


def geometry_shuffle_null(dist_stim: List[dict], syc_acts, ther_acts,
                          layer: int, n_shuffles: int = 100) -> dict:
    """Label-shuffle null for the E8/E9 geometry statistics at one layer.

    For each shuffle replicate, randomly swap syc/ther labels within each
    subtype (paired swap with Bernoulli(0.5) per pair), recompute the 12
    per-subtype directions, and measure participation ratio + mean
    off-diagonal pairwise cosine.
    """
    by_cat: Dict[str, List[int]] = {}
    for i, s in enumerate(dist_stim):
        by_cat.setdefault(s["subcategory"], []).append(i)
    cats = sorted(by_cat)
    rng = np.random.RandomState(SEED)

    # Observed (for reference)
    obs_dirs = reference.per_subtype_directions(dist_stim, syc_acts, ther_acts, [layer])
    obs_geom = reference.subtype_geometry(obs_dirs, layer, ks=(1, 2, 3, 5))
    obs_pr = obs_geom["participation_ratio"]
    cos_obs = np.array(obs_geom["pairwise_cosine"])
    n_cats = len(cats)
    off_mask = ~np.eye(n_cats, dtype=bool)
    obs_cos_mean = float(cos_obs[off_mask].mean())
    obs_cos_min = float(cos_obs[off_mask].min())

    null_prs = np.zeros(n_shuffles)
    null_cos_means = np.zeros(n_shuffles)
    null_cos_mins = np.zeros(n_shuffles)
    for b in range(n_shuffles):
        syc_shuf = list(syc_acts)
        ther_shuf = list(ther_acts)
        for cat, ix in by_cat.items():
            for i in ix:
                if rng.rand() < 0.5:
                    syc_shuf[i], ther_shuf[i] = ther_shuf[i], syc_shuf[i]
        shuf_dirs = reference.per_subtype_directions(dist_stim, syc_shuf, ther_shuf, [layer])
        shuf_geom = reference.subtype_geometry(shuf_dirs, layer, ks=(1,))
        null_prs[b] = shuf_geom["participation_ratio"]
        cm = np.array(shuf_geom["pairwise_cosine"])
        null_cos_means[b] = float(cm[off_mask].mean())
        null_cos_mins[b] = float(cm[off_mask].min())

    def one_sided_p_below(x, null):  # observed < null (smaller PR is "more structured")
        return float((null <= x).mean())

    def one_sided_p_above(x, null):  # observed > null
        return float((null >= x).mean())

    return {
        "layer": layer,
        "n_shuffles": n_shuffles,
        "observed_participation_ratio": float(obs_pr),
        "null_pr_mean": float(null_prs.mean()),
        "null_pr_std": float(null_prs.std(ddof=1)),
        # "More structured" = lower PR than null.
        "p_pr_below_null": one_sided_p_below(obs_pr, null_prs),
        "observed_cos_mean": obs_cos_mean,
        "null_cos_mean_mean": float(null_cos_means.mean()),
        "null_cos_mean_std": float(null_cos_means.std(ddof=1)),
        # "More structured" = higher mean cosine (more aligned subtypes).
        "p_cos_above_null": one_sided_p_above(obs_cos_mean, null_cos_means),
        "observed_cos_min": obs_cos_min,
        "null_cos_min_mean": float(null_cos_means.mean()),
    }


def heldout_cohens_d(syc_acts, ther_acts, layers: List[int],
                     n_fit_half: int) -> Dict[int, dict]:
    """Compute Cohen's d at each layer using a direction fit on stimuli
    [0:n_fit_half) and a test set [n_fit_half:). Reports both in-sample
    and held-out values for comparison."""
    n = len(syc_acts)
    fit_idx = list(range(n_fit_half))
    test_idx = list(range(n_fit_half, n))
    # Fit direction on first half
    d_fit = reference.contrastive_direction(
        [syc_acts[i] for i in fit_idx], [ther_acts[i] for i in fit_idx])
    # Full-set direction (in-sample reference)
    d_full = reference.contrastive_direction(syc_acts, ther_acts)
    out = {}
    for l in layers:
        dl_fit = d_fit[l]
        dl_full = d_full[l]
        # Held-out projections using d_fit
        syc_test_proj = np.array([(syc_acts[i][l] @ dl_fit).item() for i in test_idx])
        ther_test_proj = np.array([(ther_acts[i][l] @ dl_fit).item() for i in test_idx])
        sd_te = np.sqrt(((len(syc_test_proj) - 1) * syc_test_proj.var(ddof=1) +
                         (len(ther_test_proj) - 1) * ther_test_proj.var(ddof=1)) /
                        max(len(syc_test_proj) + len(ther_test_proj) - 2, 1))
        d_held = ((syc_test_proj.mean() - ther_test_proj.mean()) / sd_te
                  if sd_te > 0 else 0.0)
        # In-sample (full set projection onto full-set direction)
        syc_all_proj = np.array([(syc_acts[i][l] @ dl_full).item() for i in range(n)])
        ther_all_proj = np.array([(ther_acts[i][l] @ dl_full).item() for i in range(n)])
        sd_all = np.sqrt(((len(syc_all_proj) - 1) * syc_all_proj.var(ddof=1) +
                          (len(ther_all_proj) - 1) * ther_all_proj.var(ddof=1)) /
                         max(len(syc_all_proj) + len(ther_all_proj) - 2, 1))
        d_in_sample = ((syc_all_proj.mean() - ther_all_proj.mean()) / sd_all
                       if sd_all > 0 else 0.0)
        out[l] = {
            "cohens_d_in_sample": float(d_in_sample),
            "cohens_d_heldout": float(d_held),
            "optimism_bias": float(d_in_sample - d_held),
            "n_fit": n_fit_half,
            "n_test": len(test_idx),
        }
    return out


def main(args):
    set_seeds()
    device = reference.select_device(args.device)
    model_id = reference.MODEL_IDS[args.model]
    print(f"Device: {device}")
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # Layer grid identical to reference.py
    n_total = model.config.num_hidden_layers
    layers = (reference.pick_layers(n_total, args.n_layers)
              if args.n_layers > 0 else list(range(n_total)))
    target_layer = layers[len(layers) // 2]
    print(f"Sampled layers: {layers}   target = L{target_layer}")

    # Stimuli (match reference.py: first 100 id-sorted)
    dist_stim = sorted(
        reference.load_json(ROOT / "stimuli" / "cognitive_distortions.json"),
        key=lambda x: x["id"],
    )
    if args.n_per_cat > 0:
        dist_stim = reference.stratified_sample(dist_stim, args.n_per_cat)
    print(f"Distortion stimuli: n = {len(dist_stim)}")

    # Activations
    print("Extracting syc/ther activations ...")
    syc_acts, ther_acts = reference.extract_paired(
        model, tokenizer, dist_stim,
        "sycophantic_completion", "therapeutic_completion",
        layers, desc="syc/ther")

    results = {
        "config": {
            "model_id": model_id, "n_layers": len(layers),
            "sampled_layers": layers, "target_layer": target_layer,
            "n_distortion_stim": len(dist_stim), "seed": SEED,
        },
    }

    # M1: GroupKFold probe
    print("\n[M1] GroupKFold probe (stimulus-grouped CV)")
    groups = np.array([s["id"] for s in dist_stim])
    gkf_res = group_kfold_probe(syc_acts, ther_acts, groups, groups, layers, n_splits=5)
    stratified_res = reference.within_domain_probe(syc_acts, ther_acts, layers, cv=5)
    m1 = {"group_kfold": gkf_res, "stratified_kfold_baseline": stratified_res}
    results["M1_group_kfold_probe"] = m1
    for l in layers:
        gk = gkf_res[l]["auc_mean"]
        sk = stratified_res[l]["auc_mean"]
        print(f"  L{l}: GroupKFold AUC = {gk:.4f}   StratifiedKFold AUC = {sk:.4f}"
              f"   Δ = {gk - sk:+.4f}")

    # M2: Per-subtype label-shuffle geometry null at target_layer
    print(f"\n[M2] Per-subtype label-shuffle null at L{target_layer} "
          f"(n_shuffles = {args.n_shuffles})")
    m2 = geometry_shuffle_null(dist_stim, syc_acts, ther_acts,
                               target_layer, n_shuffles=args.n_shuffles)
    results["M2_geometry_shuffle_null"] = m2
    print(f"  Observed PR = {m2['observed_participation_ratio']:.2f}")
    print(f"  Null PR mean ± sd = {m2['null_pr_mean']:.2f} ± {m2['null_pr_std']:.2f}")
    print(f"  P(null PR ≤ observed) = {m2['p_pr_below_null']:.3f}  "
          f"(low = observed has more structure than null)")
    print(f"  Observed mean pairwise cos = {m2['observed_cos_mean']:.3f}")
    print(f"  Null cos mean = {m2['null_cos_mean_mean']:.3f} ± "
          f"{m2['null_cos_mean_std']:.3f}")
    print(f"  P(null cos_mean ≥ observed) = {m2['p_cos_above_null']:.3f}  "
          f"(low = observed has more alignment than null)")

    # M3: Held-out Cohen's d
    print(f"\n[M3] Held-out Cohen's d (fit on first 50, test on second 50)")
    n_fit_half = len(dist_stim) // 2
    m3 = heldout_cohens_d(syc_acts, ther_acts, layers, n_fit_half=n_fit_half)
    results["M3_heldout_cohens_d"] = m3
    for l in layers:
        v = m3[l]
        print(f"  L{l}: in-sample d = {v['cohens_d_in_sample']:.2f}   "
              f"held-out d = {v['cohens_d_heldout']:.2f}   "
              f"optimism = {v['optimism_bias']:+.2f}")

    out_path = RESULTS_DIR / "mitigations.json"
    reference.save_json(results, out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="1b", choices=["1b", "7b"])
    p.add_argument("--device", default=None)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-per-cat", type=int, default=0)
    p.add_argument("--n-shuffles", type=int, default=100)
    main(p.parse_args())
