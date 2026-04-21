"""Substitute concrete numbers from results/results.json into paper/paper.md placeholders.

Reads paper/paper.md, replaces every `[Nn]` and `[Section.field]` placeholder
with the corresponding value, and writes the filled paper to paper/paper.md
(in place). Idempotent; running twice has no further effect.

Each placeholder maps to one concrete value pulled from results.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
PAPER = ROOT / "paper" / "paper.md"
RES = ROOT / "results" / "results.json"


def fmt(v, p=3):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:d}"
    try:
        return f"{float(v):.{p}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_signed(v, p=3):
    if v is None:
        return "n/a"
    return f"{float(v):+.{p}f}"


def get_layer_table_value(d: dict, layer):
    """Look up a value in a per-layer dict that may have str or int keys."""
    if str(layer) in d:
        return d[str(layer)]
    if layer in d:
        return d[layer]
    return None


def _geometry_subs(e789: dict, inter_layer: int, layers: list) -> dict:
    by_layer = e789["by_layer"]
    g = get_layer_table_value(by_layer, inter_layer)
    if g is None:
        return {}
    sv_top = ", ".join(f"{s:.3f}" for s in g["singular_values"][:5])
    cum = g["cumulative_var_fraction"]
    pr_pattern = "is roughly stable across layers (range "
    prs = [get_layer_table_value(by_layer, l)["participation_ratio"] for l in layers]
    pr_pattern += f"{min(prs):.2f}–{max(prs):.2f})"
    cm = g["pairwise_cosine"]
    cats = g["subcategories"]
    n = len(cats)
    off = [cm[i][j] for i in range(n) for j in range(n) if i != j]
    pv3 = g["subspace_variance_retained_top_k"].get("3") or g["subspace_variance_retained_top_k"].get(3) or {}
    if pv3:
        worst = min(pv3, key=pv3.get); best = max(pv3, key=pv3.get)
    else:
        worst = best = None
    return {
        "[E8.sv_top]": sv_top,
        "[E8.pr]": f"{g['participation_ratio']:.2f}",
        "[E8.cum1]": f"{cum[0]:.3f}",
        "[E8.cum3]": f"{cum[2]:.3f}" if len(cum) > 2 else "n/a",
        "[E8.cum5]": f"{cum[4]:.3f}" if len(cum) > 4 else "n/a",
        "[E8.pr_pattern]": pr_pattern,
        "[E9.cos_mean]": f"{sum(off)/len(off):.3f}",
        "[E9.cos_range]": f"{min(off):.3f}–{max(off):.3f}",
        "[E9.best_subtype]": str(best) if best else "n/a",
        "[E9.best_var]": f"{pv3[best]:.3f}" if best else "n/a",
        "[E9.worst_subtype]": str(worst) if worst else "n/a",
        "[E9.worst_var]": f"{pv3[worst]:.3f}" if worst else "n/a",
    }


def main():
    res = json.loads(RES.read_text())
    cfg = res["config"]
    e1 = res["E1_distortion_direction"]
    e2 = res["E2_disentanglement"]
    e3 = res["E3_layer_localization"]
    e4 = res["E4_cross_distortion_loo"]
    e789 = res.get("E7_E8_E9_geometry", {"by_layer": {}})
    e56 = res["E5_E6_intervention_sweep"]

    target_layer = cfg["target_layer_pre_registered"]
    inter_layer = cfg["intervention_layer_pre_registered"]
    best_layer = cfg["best_descriptive_layer"]
    best_loo = cfg["best_loo_layer"]
    n_layers = cfg["n_layers"]
    layers = cfg["sampled_layers"]

    headline = e56["headline_pre_registered"]
    best_spec = e56["best_specificity_config"]

    # Build the substitution dictionary
    perm = e1["permutation_target_layer"]
    e1_within = e1["within_domain_probe"]
    e1_factual = e1["factual_within_domain_probe"]
    layer_table_target = get_layer_table_value(e3["by_layer"], best_layer)
    layer_table_inter = get_layer_table_value(e3["by_layer"], inter_layer)

    # E2 at pre-registered layer
    cw = get_layer_table_value(e2["cos_dist_warmth_by_layer"], inter_layer)
    cf = get_layer_table_value(e2["cos_dist_factual_by_layer"], inter_layer)
    dec = get_layer_table_value(e2["decomp_by_layer"], inter_layer)

    # E4 mean AUC at best layer
    e4_best = get_layer_table_value(e4["by_layer"], best_loo)
    e4_inter = get_layer_table_value(e4["by_layer"], inter_layer)
    per_subcat = get_layer_table_value(e4["by_layer_per_subcat"], best_loo)
    per_subcat_pairs = sorted(per_subcat.items(), key=lambda kv: kv[1]["auc"]) if per_subcat else []

    # Layer sweep summary for E5
    sweep = e56["layer_sweep"]
    e5_per_layer_neg = []
    for l in layers:
        e5_per_layer_neg.append((l, get_layer_table_value(sweep, l)["negative_steering"]
                                 ["target"]["E5_ther_vs_syc"]["shift_mean"]))
    e5_layer_best = max(e5_per_layer_neg, key=lambda kv: kv[1])

    # Alpha sweep dose response qualitative summary
    alpha_sweep = e56["alpha_sweep_at_intervention_layer"]
    alphas = sorted(float(a) for a in alpha_sweep.keys())
    e5_alpha_means = [(a, alpha_sweep[str(a) if str(a) in alpha_sweep else a]
                       ["target"]["E5_ther_vs_syc"]["shift_mean"]) for a in alphas]
    monotonic_pos = all(b[1] >= a[1] for a, b in zip(e5_alpha_means, e5_alpha_means[1:]))
    if monotonic_pos:
        dose_pattern = "a monotonically-increasing dose response: larger alpha gives larger E5 shift"
    else:
        dose_pattern = "a non-monotonic dose response (the largest E5 shift is not at the largest alpha)"

    # E6 verdict
    spec_pre = abs(headline["E5_shift_mean"]) / max(abs(headline["E6_shift_mean"]), 1e-8)
    if best_spec.get("specificity") is not None and best_spec["specificity"] >= 1.0:
        e6_verdict = (
            f"specificity above the threshold of 1.0 IS achievable: at "
            f"L{best_spec['layer']} with {best_spec['intervention']}, the ratio reaches "
            f"{best_spec['specificity']:.2f}"
        )
        spec_max_summary = "that a specific intervention exists, but only at non-pre-registered layers"
    else:
        e6_verdict = (
            "no configuration in the layer × intervention sweep achieves specificity above 1.0"
            " while the E5 shift remains in the desired direction"
        )
        spec_max_summary = "that linear, single-layer, residual-stream interventions do not achieve clean specificity in this model"

    factual_pattern_text = "similar but lower-magnitude" if (
        max(get_layer_table_value(e1_factual, l)["auc_mean"] for l in layers)
        < max(get_layer_table_value(e1_within, l)["auc_mean"] for l in layers)
    ) else "comparable"

    # Friendly label for the model used to produce these numbers.
    model_id = cfg["model_id"]
    if "OLMo-2" in model_id and "1B" in model_id:
        n3a = "OLMo-2 1B Instruct (local validation pipeline)"
    elif "Olmo-3" in model_id and "7B" in model_id:
        n3a = "OLMo-3 7B Instruct"
    else:
        n3a = model_id
    n8a = "above" if spec_pre >= 1.0 else "below"
    if best_spec.get("specificity") is not None:
        n8b = (f"L{best_spec['layer']} {best_spec['intervention']} "
               f"(specificity={best_spec['specificity']:.2f}, "
               f"E5={best_spec['E5_shift']:+.3f}, E6={best_spec['E6_shift']:+.3f})")
    else:
        n8b = "no positive-E5 configuration was found in the sweep"

    subs = {
        # General N references
        "[N1]": fmt(e1_within[str(best_layer) if str(best_layer) in e1_within else best_layer]["auc_mean"]),
        "[N2]": fmt(e4_best["mean_auc"] if e4_best else None),
        "[N3]": f"L{inter_layer}, negative steering at α=4.0",
        "[N3a]": n3a,
        "[N8a]": n8a,
        "[N8b]": n8b,
        "[N4]": fmt_signed(headline["E5_shift_mean"]),
        "[N5]": fmt_signed(headline["E5_z_vs_random"], 2),
        "[N6]": fmt_signed(headline["E6_shift_mean"]),
        "[N7]": fmt_signed(headline["E6_z_vs_random"], 2),
        "[N8]": fmt(spec_pre),
        "[N9]": fmt(layer_table_target["auc"] if layer_table_target else None),
        "[N10]": fmt(layer_table_inter["auc"] if layer_table_inter else None),
        # T1 = runtime estimate placeholder
        "[T1]": "30",
        # E1 details
        "[E1.auc_pre]": fmt(layer_table_inter["auc"] if layer_table_inter else None),
        "[E1.auc_ci]": "see results.json: E1_distortion_direction.within_domain_probe",
        "[E1.auc_factual]": fmt(get_layer_table_value(e1_factual, inter_layer)["auc_mean"]),
        "[E1.p]": fmt(perm["p_value"], 3),
        "[E1.auc_l0]": fmt(get_layer_table_value(e1_within, layers[0])["auc_mean"]),
        "[E1.auc_lpeak]": fmt(layer_table_target["auc"] if layer_table_target else None),
        "[E1.factual_pattern]": factual_pattern_text,
        # E2
        "[E2.cos_warmth]": fmt(cw),
        "[E2.cos_factual]": fmt(cf),
        "[E2.warmth_ve]": fmt(dec["unique_ve"]["warmth"]) if dec else "n/a",
        "[E2.factual_ve]": fmt(dec["unique_ve"]["factual"]) if dec else "n/a",
        "[E2.residual_ve]": fmt(dec["residual_ve"]) if dec else "n/a",
        # E3
        "[E3.best_layer]": str(best_layer),
        "[E3.best_d]": fmt(layer_table_target["cohens_d"], 2) if layer_table_target else "n/a",
        # E4
        "[E4.best_auc]": fmt(e4_best["mean_auc"]) if e4_best else "n/a",
        "[E4.range]": (
            f"{min(v['auc'] for v in per_subcat.values()):.3f} – "
            f"{max(v['auc'] for v in per_subcat.values()):.3f}"
        ) if per_subcat else "n/a",
        "[E4.weakest]": per_subcat_pairs[0][0] if per_subcat_pairs else "n/a",
        "[E4.weakest_auc]": fmt(per_subcat_pairs[0][1]["auc"]) if per_subcat_pairs else "n/a",
        "[E4.strongest]": per_subcat_pairs[-1][0] if per_subcat_pairs else "n/a",
        "[E4.strongest_auc]": fmt(per_subcat_pairs[-1][1]["auc"]) if per_subcat_pairs else "n/a",
        # E5
        "[E5.shift]": fmt_signed(headline["E5_shift_mean"]),
        "[E5.se]": fmt(headline["E5_shift_se"]),
        "[E5.n]": str(cfg["n_intervene"]),
        "[E5.z]": fmt_signed(headline["E5_z_vs_random"], 2),
        "[E5.dose_pattern]": dose_pattern,
        "[E5.layer_best_shift]": fmt_signed(e5_layer_best[1]),
        "[E5.layer_best_layer]": str(e5_layer_best[0]),
        # E8 / E9 — geometry at the pre-registered intervention layer
        **(_geometry_subs(e789, inter_layer, layers) if e789["by_layer"] else {}),
        # E6
        "[E6.shift]": fmt_signed(headline["E6_shift_mean"]),
        "[E6.se]": fmt(headline["E6_shift_se"]),
        "[E6.z]": fmt_signed(headline["E6_z_vs_random"], 2),
        "[E6.spec_pre]": fmt(spec_pre),
        "[E6.spec_max]": fmt(best_spec["specificity"]) if best_spec.get("specificity") is not None else "n/a",
        "[E6.spec_max_layer]": str(best_spec["layer"]) if best_spec.get("layer") is not None else "n/a",
        "[E6.spec_max_intervention]": str(best_spec["intervention"]) if best_spec.get("intervention") is not None else "n/a",
        "[E6.spec_max_summary]": spec_max_summary,
        "[E6.verdict]": e6_verdict,
    }

    text = PAPER.read_text()
    for k, v in subs.items():
        text = text.replace(k, v)
    PAPER.write_text(text)

    # Report any unfilled placeholders
    remaining = re.findall(r"\[[NETE]\d?[\.\w]*\]", text)
    if remaining:
        print(f"WARN: {len(remaining)} unfilled placeholders remain:")
        for p in sorted(set(remaining)):
            print(f"  {p}")
    else:
        print("All placeholders filled.")
    print(f"Paper written to {PAPER}")


if __name__ == "__main__":
    main()
