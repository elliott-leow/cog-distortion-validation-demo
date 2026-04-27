"""Generate presentation graphs from trial-2 results (OLMo-3-7B-Instruct)."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

HERE = Path(__file__).resolve().parent
RES_PATH = HERE.parent / "results(2).json"
with RES_PATH.open() as f:
    D = json.load(f)

CFG = D["config"]
LAYERS = CFG["sampled_layers"]
TARGET = CFG["target_layer_pre_registered"]       # 16
BEST_DESC = CFG["best_descriptive_layer"]          # 14
N_LAYERS = CFG["n_layers"]                         # 32
MODEL = CFG["model_id"]

SYC_COLOR = "#d94a4a"
THER_COLOR = "#2a8dbd"
WARM_COLOR = "#f0a93b"
FACT_COLOR = "#6b4aa8"
NEUTRAL = "#666666"


def L(key_dict):
    """Yield (layer_int, value) pairs in layer order."""
    out = []
    for k in key_dict:
        try:
            out.append((int(k), key_dict[k]))
        except (TypeError, ValueError):
            pass
    return sorted(out)


# ---------------------------------------------------------------------------
# Figure 1 — Where the distortion direction lives: probe AUC + Cohen's d
# ---------------------------------------------------------------------------
def fig1_localization():
    e1 = dict(L(D["E1_distortion_direction"]["within_domain_probe"]))
    e1_fact = dict(L(D["E1_distortion_direction"]["factual_within_domain_probe"]))
    e3 = dict(L(D["E3_layer_localization"]["by_layer"]))

    layers = sorted(e3.keys())
    auc = [e1[l]["auc_mean"] for l in layers]
    auc_fact = [e1_fact[l]["auc_mean"] for l in layers]
    d_cohen = [e3[l]["cohens_d"] for l in layers]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(layers, auc, "o-", color=SYC_COLOR, lw=2,
             label="Distortion probe AUC (syc vs ther)")
    ax1.plot(layers, auc_fact, "s--", color=FACT_COLOR, lw=1.5, alpha=0.7,
             label="Factual-sycophancy probe AUC")
    ax1.set_ylim(0.45, 1.03)
    ax1.set_ylabel("Probe AUC (5-fold CV)", color=SYC_COLOR)
    ax1.tick_params(axis="y", labelcolor=SYC_COLOR)
    ax1.axhline(0.5, color=NEUTRAL, ls=":", lw=1, alpha=0.7)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(layers, d_cohen, "D-", color=THER_COLOR, lw=2,
             label="Cohen's d (syc − ther projection)")
    ax2.set_ylabel("Cohen's d", color=THER_COLOR)
    ax2.tick_params(axis="y", labelcolor=THER_COLOR)

    ax1.axvline(TARGET, color="black", ls="-", lw=1, alpha=0.4)
    ax1.text(TARGET + 0.3, 0.55, f"pre-registered\nlayer {TARGET}",
             fontsize=9, alpha=0.7)
    ax1.axvline(BEST_DESC, color="green", ls="--", lw=1, alpha=0.6)
    ax1.text(BEST_DESC - 6.5, 0.98, f"best Cohen's d: L{BEST_DESC}",
             fontsize=9, color="green", alpha=0.8)

    ax1.set_xlabel("Residual-stream layer")
    fig.suptitle("E1/E3 — distortion direction separates from layer 0 onward\n"
                 f"({MODEL}, 100 syc + 100 ther stimuli)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig1_probe_auc_and_cohens_d.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Disentanglement: how much of d_dist is warmth vs factual?
# Surprise: factual direction eats ~12x more unique VE than warmth
# ---------------------------------------------------------------------------
def fig2_disentanglement():
    dec = dict(L(D["E2_disentanglement"]["decomp_by_layer"]))
    layers = sorted(dec.keys())

    warm_unique = [dec[l]["unique_ve"]["warmth"] for l in layers]
    fact_unique = [dec[l]["unique_ve"]["factual"] for l in layers]
    residual = [dec[l]["residual_ve"] for l in layers]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.array(layers)
    width = 1.1

    ax.bar(x - width / 2, warm_unique, width=width, color=WARM_COLOR,
           label="Unique VE: warmth axis")
    ax.bar(x + width / 2, fact_unique, width=width, color=FACT_COLOR,
           label="Unique VE: factual-syc axis")
    ax.plot(x, residual, "o-", color=NEUTRAL, lw=1.5, alpha=0.6,
            label="Residual VE (unexplained)")

    ax.set_xlabel("Residual-stream layer")
    ax.set_ylabel("Variance of $d_{\\mathrm{dist}}$ explained")
    ax.set_ylim(0, 1.0)
    ax.axvline(BEST_DESC, color="green", ls="--", lw=1, alpha=0.6)

    # Annotate the "surprise" at best descriptive layer
    w14 = dec[BEST_DESC]["unique_ve"]["warmth"]
    f14 = dec[BEST_DESC]["unique_ve"]["factual"]
    ax.annotate(
        f"L{BEST_DESC}: factual unique VE = {f14:.2f}\n"
        f"warmth unique VE = {w14:.3f}\n"
        f"→ factual ~{f14/max(w14,1e-6):.0f}× warmth",
        xy=(BEST_DESC + 0.4, f14),
        xytext=(BEST_DESC - 9, 0.55),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.6),
    )

    ax.set_title("E2 — distortion direction is nearly orthogonal to warmth,\n"
                 "but shares substantial variance with factual-sycophancy")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_disentanglement.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Cross-distortion leave-one-out generalization (E4)
# Surprise: AUC = 1.0 for all 12 subtypes, even at layer 0
# ---------------------------------------------------------------------------
def fig3_loo_generalization():
    per = dict(L(D["E4_cross_distortion_loo"]["by_layer_per_subcat"]))
    subcats = list(per[0].keys())
    # layers to display: a subset
    layers_show = [l for l in sorted(per.keys()) if l in (0, 8, 14, 16, 22, 31)]

    acc_matrix = np.array([[per[l][s]["acc"] for s in subcats] for l in layers_show])

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    im = ax.imshow(acc_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(subcats)))
    ax.set_xticklabels([s.replace("_", "\n") for s in subcats],
                       rotation=0, fontsize=8)
    ax.set_yticks(range(len(layers_show)))
    ax.set_yticklabels([f"L{l}" for l in layers_show])
    for i in range(len(layers_show)):
        for j in range(len(subcats)):
            ax.text(j, i, f"{acc_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if acc_matrix[i, j] > 0.7 else "white")

    fig.colorbar(im, ax=ax, label="Held-out accuracy")
    ax.set_title("E4 — leave-one-out generalization across 12 distortion subtypes\n"
                 "(train on 11, test on held-out subtype; AUC = 1.00 everywhere)")
    ax.set_xlabel("Held-out distortion subtype")
    ax.set_ylabel("Layer")
    fig.tight_layout()
    fig.savefig(HERE / "fig3_loo_subcategory_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Intervention headline (E5/E6): steering shifts preference
# ---------------------------------------------------------------------------
def fig4_intervention_headline():
    hd = D["E5_E6_intervention_sweep"]["headline_pre_registered"]
    ls16 = D["E5_E6_intervention_sweep"]["layer_sweep"]["16"]["negative_steering"]

    # E5 target shift vs random-direction null
    E5_shift = hd["E5_shift_mean"]
    E5_se = hd["E5_shift_se"]
    E6_shift = hd["E6_shift_mean"]
    E6_se = hd["E6_shift_se"]
    rand_E5 = np.array(ls16["random_shifts"]["E5_ther_vs_syc"])
    rand_E6 = np.array(ls16["random_shifts"]["E6_ther_vs_cold"])
    ci_lo, ci_hi = hd["specificity_ratio_bootstrap_95_ci"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # Left: bar chart — target vs random nulls
    ax = axes[0]
    labels = ["E5:  ther-vs-syc\n(positive = more therapeutic)",
              "E6:  ther-vs-cold\n(positive = warmer; 0 = specific)"]
    target_vals = [E5_shift, E6_shift]
    target_err = [E5_se, E6_se]
    rand_mean = [rand_E5.mean(), rand_E6.mean()]
    rand_std = [rand_E5.std(ddof=1), rand_E6.std(ddof=1)]

    x = np.arange(2)
    ax.bar(x - 0.2, target_vals, width=0.38, yerr=target_err,
           color=[SYC_COLOR, WARM_COLOR], label="d_dist steering", capsize=4)
    ax.bar(x + 0.2, rand_mean, width=0.38, yerr=rand_std,
           color="lightgray", label="Random-direction null", capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Log-prob shift (Δ)")
    ax.set_title(f"E5/E6 — pre-registered intervention\n"
                 f"layer {hd['layer']}, α={hd['alpha']}, {hd['intervention']}")
    ax.text(0 - 0.2, E5_shift + 0.02,
            f"z={hd['E5_z_vs_random']:.1f}σ", ha="center", fontsize=9)
    ax.text(1 - 0.2, E6_shift - 0.04,
            f"z={hd['E6_z_vs_random']:.1f}σ", ha="center", fontsize=9)
    ax.legend(fontsize=9, loc="upper right")

    # Right: specificity ratio bootstrap
    ax = axes[1]
    spec = hd["specificity_ratio"]
    spec_mean = hd["specificity_ratio_bootstrap_mean"]
    ax.errorbar([spec], [0], xerr=[[spec - ci_lo], [ci_hi - spec]],
                fmt="D", color=SYC_COLOR, capsize=8, ms=10, lw=2,
                label=f"specificity ratio = {spec:.2f}\n95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(1.0, 0.2, "  ratio = 1\n  (no specificity)", fontsize=9, alpha=0.7)
    ax.set_xlim(0, max(ci_hi * 1.15, 4))
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("|E5 shift| / |E6 shift|   (signed specificity)")
    ax.set_title("E6 — specificity ratio (5,000 bootstrap resamples)\n"
                 "Surprise: steering hits both therapeutic and warmth axes")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(HERE / "fig4_intervention_headline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Alpha-dose response
# ---------------------------------------------------------------------------
def fig5_alpha_sweep():
    sweep = D["E5_E6_intervention_sweep"]["alpha_sweep_at_intervention_layer"]
    alphas = sorted(float(a) for a in sweep)
    e5 = [sweep[str(a)]["target"]["E5_ther_vs_syc"]["shift_mean"] for a in alphas]
    e5_se = [sweep[str(a)]["target"]["E5_ther_vs_syc"]["shift_se"] for a in alphas]
    e6 = [sweep[str(a)]["target"]["E6_ther_vs_cold"]["shift_mean"] for a in alphas]
    e6_se = [sweep[str(a)]["target"]["E6_ther_vs_cold"]["shift_se"] for a in alphas]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.errorbar(alphas, e5, yerr=e5_se, fmt="o-", color=SYC_COLOR, lw=2,
                capsize=3, label="E5: ther-vs-syc shift (target)")
    ax.errorbar(alphas, e6, yerr=e6_se, fmt="s--", color=WARM_COLOR, lw=2,
                capsize=3, label="E6: ther-vs-cold shift (collateral)")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Steering magnitude α (log scale)")
    ax.set_ylabel("Log-prob shift (Δ)")
    ax.set_title(f"E5/E6 — dose-response at intervention layer {TARGET}\n"
                 "Therapeutic gain scales with α; warmth cost also rises")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "fig5_alpha_sweep.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Per-layer intervention sweep (E5/E6 as a function of layer)
# Surprise: earlier layers give much better specificity
# ---------------------------------------------------------------------------
def fig6_layer_sweep():
    sweep = D["E5_E6_intervention_sweep"]["layer_sweep"]
    layers = sorted(int(l) for l in sweep)

    def get(l, mode, metric):
        return sweep[str(l)][mode]["target"][metric]["shift_mean"]

    def get_se(l, mode, metric):
        return sweep[str(l)][mode]["target"][metric]["shift_se"]

    e5_ns = [get(l, "negative_steering", "E5_ther_vs_syc") for l in layers]
    e6_ns = [get(l, "negative_steering", "E6_ther_vs_cold") for l in layers]
    e5_ab = [get(l, "ablation", "E5_ther_vs_syc") for l in layers]
    e6_ab = [get(l, "ablation", "E6_ther_vs_cold") for l in layers]
    e5_ns_se = [get_se(l, "negative_steering", "E5_ther_vs_syc") for l in layers]
    e6_ns_se = [get_se(l, "negative_steering", "E6_ther_vs_cold") for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)

    ax = axes[0]
    ax.errorbar(layers, e5_ns, yerr=e5_ns_se, fmt="o-", color=SYC_COLOR,
                lw=2, capsize=3, label="E5 (target)")
    ax.errorbar(layers, e6_ns, yerr=e6_ns_se, fmt="s--", color=WARM_COLOR,
                lw=2, capsize=3, label="E6 (warmth collateral)")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(TARGET, color="green", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel("Log-prob shift (Δ)")
    ax.set_title(f"Negative steering (α={CFG['alpha_layer_sweep']}) by layer")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(layers, e5_ab, "o-", color=SYC_COLOR, lw=2, label="E5 (target)")
    ax.plot(layers, e6_ab, "s--", color=WARM_COLOR, lw=2, label="E6 (warmth collateral)")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel("Log-prob shift (Δ)")
    ax.set_title("Ablation (project out $d_{\\mathrm{dist}}$) by layer")
    ax.legend(fontsize=9)

    # Annotate best specificity (meaningful)
    bm = D["E5_E6_intervention_sweep"]["best_specificity_config_meaningful"]
    ax.annotate(
        f"best meaningful\nspecificity:\nL{bm['layer']} {bm['intervention']}\n"
        f"E5={bm['E5_shift']:.2f}, E6={bm['E6_shift']:.3f}\n"
        f"ratio={bm['specificity']:.1f}",
        xy=(bm["layer"], bm["E5_shift"]),
        xytext=(bm["layer"] + 3, bm["E5_shift"] + 0.05),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", alpha=0.6),
    )

    fig.suptitle("E5/E6 — intervention layer sweep: negative steering vs ablation",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(HERE / "fig6_layer_sweep.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Geometry: SVD spectrum + participation ratio by layer
# ---------------------------------------------------------------------------
def fig7_geometry_spectrum():
    geo = dict(L(D["E7_E8_E9_geometry"]["by_layer"]))
    layers = sorted(geo.keys())
    pr = [geo[l]["participation_ratio"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))

    ax = axes[0]
    # Stacked variance fractions at best descriptive layer
    g = geo[BEST_DESC]
    vf = g["var_fraction"]
    ax.bar(range(1, len(vf) + 1), vf, color=SYC_COLOR)
    for i, v in enumerate(vf):
        if v > 0.03:
            ax.text(i + 1, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Variance fraction")
    ax.set_title(f"E7 — SVD spectrum of 12 per-subtype directions at L{BEST_DESC}\n"
                 f"PC1 alone = {vf[0]*100:.1f}% of variance")

    ax = axes[1]
    ax.plot(layers, pr, "o-", color=FACT_COLOR, lw=2)
    ax.axhline(1, color="black", ls=":", lw=1, alpha=0.6, label="PR=1 (pure 1-D)")
    ax.axhline(12, color="gray", ls=":", lw=1, alpha=0.6, label="PR=12 (isotropic)")
    ax.set_xlabel("Residual-stream layer")
    ax.set_ylabel("Participation ratio")
    ax.set_title("E8 — effective dimensionality of the direction set\n"
                 f"min PR across layers = {min(pr):.2f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(HERE / "fig7_geometry.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — Pairwise cosines between per-subtype directions at best layer
# ---------------------------------------------------------------------------
def fig8_pairwise_cosines():
    g = D["E7_E8_E9_geometry"]["by_layer"][str(BEST_DESC)]
    subcats = g["subcategories"]
    C = np.array(g["pairwise_cosine"])

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(C, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(len(subcats)))
    ax.set_xticklabels(subcats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(subcats)))
    ax.set_yticklabels(subcats, fontsize=8)
    off = C[~np.eye(len(C), dtype=bool)]
    for i in range(len(subcats)):
        for j in range(len(subcats)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if C[i, j] < 0.3 or C[i, j] > 0.85 else "black")
    fig.colorbar(im, ax=ax, label="cos(d_subtype_i, d_subtype_j)")
    ax.set_title(f"E9 — pairwise cosine between 12 per-subtype directions at L{BEST_DESC}\n"
                 f"off-diagonal mean = {off.mean():.2f}  (identical direction → 1.0)")
    fig.tight_layout()
    fig.savefig(HERE / "fig8_pairwise_cosines.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary text (printed + saved)
# ---------------------------------------------------------------------------
def write_summary():
    hd = D["E5_E6_intervention_sweep"]["headline_pre_registered"]
    bm = D["E5_E6_intervention_sweep"]["best_specificity_config_meaningful"]
    dec14 = D["E2_disentanglement"]["decomp_by_layer"][str(BEST_DESC)]
    e3_14 = D["E3_layer_localization"]["by_layer"][str(BEST_DESC)]
    perm = D["E1_distortion_direction"]["permutation_target_layer"]
    geo14 = D["E7_E8_E9_geometry"]["by_layer"][str(BEST_DESC)]

    lines = [
        f"Trial 2 — {MODEL} — summary",
        "",
        f"E1 (layer {perm['layer']}): probe AUC = {perm['observed_auc']:.2f} vs null "
        f"mean {perm['null_mean_auc']:.2f} ± {perm['null_std_auc']:.3f}, p = {perm['p_value']:.3f}",
        f"E3 (layer {BEST_DESC}): Cohen's d = {e3_14['cohens_d']:.2f}, AUC = {e3_14['auc']:.2f}",
        "",
        f"E2 disentanglement at layer {BEST_DESC}:",
        f"  unique VE warmth = {dec14['unique_ve']['warmth']:.3f}",
        f"  unique VE factual = {dec14['unique_ve']['factual']:.3f}   "
        f"(~{dec14['unique_ve']['factual']/max(dec14['unique_ve']['warmth'],1e-6):.0f}× warmth)",
        f"  residual VE     = {dec14['residual_ve']:.3f}",
        "",
        f"E5/E6 headline (layer {hd['layer']}, {hd['intervention']}, α={hd['alpha']}):",
        f"  E5 therapeutic shift = {hd['E5_shift_mean']:+.3f} ± {hd['E5_shift_se']:.3f}  "
        f"(z = {hd['E5_z_vs_random']:.1f} vs random)",
        f"  E6 warmth collateral = {hd['E6_shift_mean']:+.3f} ± {hd['E6_shift_se']:.3f}  "
        f"(z = {hd['E6_z_vs_random']:.1f} vs random)",
        f"  specificity ratio = {hd['specificity_ratio']:.2f}  95% CI "
        f"[{hd['specificity_ratio_bootstrap_95_ci'][0]:.2f}, "
        f"{hd['specificity_ratio_bootstrap_95_ci'][1]:.2f}]",
        "",
        f"Best meaningful specificity: L{bm['layer']} {bm['intervention']}, "
        f"E5={bm['E5_shift']:.3f}, E6={bm['E6_shift']:.3f}, ratio={bm['specificity']:.1f}",
        "",
        f"Geometry at L{BEST_DESC}: PC1 = {geo14['var_fraction'][0]*100:.1f}%, "
        f"PR = {geo14['participation_ratio']:.2f} / 12 subtypes",
    ]
    text = "\n".join(lines)
    print(text)
    (HERE / "_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    fig1_localization()
    fig2_disentanglement()
    fig3_loo_generalization()
    fig4_intervention_headline()
    fig5_alpha_sweep()
    fig6_layer_sweep()
    fig7_geometry_spectrum()
    fig8_pairwise_cosines()
    write_summary()
    print("Wrote graphs to", HERE)
