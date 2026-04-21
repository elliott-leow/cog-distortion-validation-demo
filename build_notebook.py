"""Build notebook.ipynb from reference.py for OLMo-3 7B Colab execution.

Produces a single self-contained .ipynb with embedded stimuli files so it can
be opened in Colab and run end-to-end without external downloads.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REF = ROOT / "reference.py"
OUT = ROOT / "notebook.ipynb"
STIM_DIR = ROOT / "stimuli"

# stimuli embedded inline (only the ones reference.py loads)
EMBED = ["cognitive_distortions.json"]


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def factual_subset_str(n: int = 100) -> str:
    """Subset v2_factual_control.json to first n items, return as compact JSON string."""
    fact = json.loads((STIM_DIR / "v2_factual_control.json").read_text())
    fact_sub = sorted(fact, key=lambda x: x["id"])[:n]
    return json.dumps(fact_sub)


def _patch_root(src: str) -> str:
    """Replace the module-level `ROOT = Path(__file__).parent` with a
    notebook-safe equivalent. `__file__` is undefined inside a Jupyter cell.
    """
    return src.replace(
        "ROOT = Path(__file__).parent",
        "ROOT = Path(globals().get('__file__', '.')).resolve().parent if globals().get('__file__') else Path.cwd()",
    )


def parse_reference() -> str:
    """Load reference.py source, drop the __main__ guard so we can drive run()
    explicitly from the notebook with custom args.

    Tolerant of single/double quotes and minor whitespace variation in the
    guard line; fails loud if the guard is missing so notebook breakage
    surfaces at build time rather than at Colab execution time.
    """
    src = REF.read_text()
    m = re.search(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""", src, re.MULTILINE)
    if m is None:
        raise RuntimeError(
            "build_notebook.parse_reference: could not find `if __name__ == \"__main__\":` "
            "guard in reference.py. If you renamed or removed it, update parse_reference."
        )
    return _patch_root(src[:m.start()].rstrip() + "\n")


def build():
    cells = []

    cells.append(md("""# Mechanistic Interpretability of Cognitive-Distortion Validation

Self-contained Colab notebook reproducing all six experiments on **OLMo-3 7B Instruct**.

**Runtime:** ~30 minutes on a single A100 (40 GB).
**Author:** Elliott Leow

The notebook is the same code as the local `reference.py` script (which validates on OLMo-2 1B in ~5 min on Apple MPS), with the model id, dtype, and stimulus counts adapted for the 7B Colab run. Stimuli are embedded inline so no external download is required.
"""))

    cells.append(md("## 1. Install dependencies"))
    cells.append(code(
        "# OLMo-3 requires transformers >= 4.57. We don't upgrade numpy/scipy/sklearn/\n"
        "# matplotlib/tqdm because Colab pins them for numba and tensorflow.\n"
        "%pip install -q \"transformers>=4.57.0\" accelerate\n"
    ))

    cells.append(md("## 2. Embed stimuli\n\n"
                    "Two stimulus files are written to disk so the rest of the code can `load_json` them. "
                    "The cognitive-distortion file is from the [clinical_sycophancy_demo](https://github.com/Elliott-Leow/clinical_sycophancy_demo) repo (100 items, 12 distortion subtypes, three completion types each). "
                    "The factual-control file is a 100-item subset of the v2 factual_control set."))
    cells.append(code(
        "import json, os\n"
        "os.makedirs('stimuli', exist_ok=True)\n"
    ))
    for fname in EMBED:
        body = (STIM_DIR / fname).read_text()
        # Use json.dumps + json.loads round-trip with compact form to control quoting
        compact = json.dumps(json.loads(body))
        cells.append(code(
            f"# {fname}: {len(json.loads(body))} items\n"
            f"_data = {compact!r}\n"
            f"open('stimuli/{fname}', 'w').write(_data)\n"
        ))
    cells.append(code(
        "# v2_factual_control.json: 100-item subset (id-sorted) for matched factual baseline\n"
        f"_fact = {factual_subset_str(100)!r}\n"
        "open('stimuli/v2_factual_control.json', 'w').write(_fact)\n"
        "print('stimuli/:', sorted(os.listdir('stimuli')))\n"
    ))

    cells.append(md("## 3. Pipeline source\n\nThe full `reference.py` is included verbatim below — every helper, hook, and pipeline function. The `__main__` guard is stripped so `run(args)` can be invoked from the notebook with custom Colab settings."))
    cells.append(code(parse_reference()))

    cells.append(md("## 4. Configure and run\n\n"
                    "- `model='7b'` selects `allenai/Olmo-3-7B-Instruct`.\n"
                    "- `n_per_cat=0` and `n_intervene=50` use the full 100-item distortion set for direction extraction and a 50-stimulus subset for the intervention sweep.\n"
                    "- `n_random=10` random-direction controls per (layer, intervention) configuration.\n"
                    "- `n_layers=16` samples 16-of-32 layers (every other) to keep the layer sweep under ~45 min on A100; full-32 mode is available by setting `n_layers=0`.\n"
                    "- `alpha=4.0` is the headline negative-steering magnitude (the full alpha sweep also runs)."))
    cells.append(code(
        "import argparse\n"
        "args = argparse.Namespace(\n"
        "    model='7b', device=None,\n"
        "    n_per_cat=0, n_intervene=50, n_random=10,\n"
        "    n_random_headline=30, n_perms=500, n_layers=16, alpha=4.0,\n"
        "    quick=False,\n"
        ")\n"
        "run(args)\n"
    ))

    cells.append(md("## 5. View results\n\n"
                    "`run(args)` has already written `results/results.json` and the PNG figures to "
                    "`figures/`. Both are persisted on the Colab filesystem; download them from the "
                    "file browser or run the cell below to copy `results.json` to your Google Drive."))
    cells.append(code(
        "import json, os\n"
        "assert os.path.exists('results/results.json'), \\\n"
        "    'results.json missing — did run() finish?'\n"
        "results = json.load(open('results/results.json'))\n"
        "print('=== config ===')\n"
        "for k, v in results['config'].items():\n"
        "    print(f'  {k}: {v}')\n"
        "print('\\n=== E1 (within-domain probe) ===')\n"
        "perm = results['E1_distortion_direction']['permutation_target_layer']\n"
        "print(f\"  L{perm['layer']} within-domain probe AUC = {perm['observed_auc']:.3f}; \"\n"
        "      f\"null mean = {perm['null_mean_auc']:.3f}; p = {perm['p_value']:.3f}\")\n"
        "print('\\n=== E5/E6 headline (held-out direction, n=30 random null) ===')\n"
        "h = results['E5_E6_intervention_sweep']['headline_pre_registered']\n"
        "for k, v in h.items():\n"
        "    print(f'  {k}: {v}')\n"
        "print('\\n=== best specificity config ===')\n"
        "print(results['E5_E6_intervention_sweep']['best_specificity_config'])\n"
    ))
    cells.append(md("**Optional: persist results.json to Google Drive.** "
                    "Uncomment the cell below to mount Drive and copy the full results file + figures — "
                    "the Colab ephemeral filesystem is wiped when the runtime disconnects."))
    cells.append(code(
        "# from google.colab import drive\n"
        "# drive.mount('/content/drive')\n"
        "# import shutil, os\n"
        "# out_dir = '/content/drive/MyDrive/clinical_sycophancy_7b_run'\n"
        "# os.makedirs(out_dir, exist_ok=True)\n"
        "# shutil.copy('results/results.json', out_dir)\n"
        "# shutil.copytree('figures', f'{out_dir}/figures', dirs_exist_ok=True)\n"
        "# print('Saved to', out_dir)\n"
    ))

    cells.append(md("## 6. Figures\n\n"
                    "The nine figures below are generated by `make_figures(...)` inside `run()` and "
                    "saved to `figures/`. Each one is captioned inline so the notebook remains "
                    "readable without the paper."))
    # Per-figure captions that make the notebook readable standalone
    captions = {
        "fig1_layer_auc.png":
            "**Fig. 1 — E1 within-domain probe AUC per layer.** Saturated near 1.000 across layers; "
            "see the style-confound caveat in §4.1 and the mitigations (M1 GroupKFold) in §4.10.",
        "fig2_cosine.png":
            "**Fig. 2 — E2 cosine similarity of `d_dist` with `d_warmth` and `d_factual`.** "
            "At the prespecified layer, cos(d_dist, d_warmth) ≈ 0.16, cos(d_dist, d_factual) ≈ 0.47.",
        "fig3_decomposition.png":
            "**Fig. 3 — E2 Gram-Schmidt decomposition (warmth-first ordering).** Unique variance in "
            "`d_dist` attributable to warmth, factual, and residual. Factual-first ordering is in the "
            "JSON.",
        "fig4a_loo_by_layer.png":
            "**Fig. 4a — E4 cross-distortion LOO mean AUC per layer.** Saturated at 1.000; see the "
            "§4.4 saturation caveat.",
        "fig4b_loo_by_subcat.png":
            "**Fig. 4b — E4 per-subtype held-out AUC at the best layer.** All 12 subtypes saturate; "
            "no meaningful per-subtype ranking under saturation (see §4.4 caveat).",
        "fig5_layer_sweep.png":
            "**Fig. 5 — E5/E6 intervention shift across all sampled layers.** Left panel: projection-"
            "ablation. Right panel: negative steering (α=4.0). Red = E5 (ther vs syc, target). "
            "Blue = E6 (ther vs cold, off-target). Dotted lines are random-direction nulls (not on "
            "directly comparable scales across panels — see §4.8).",
        "fig6_alpha_sweep.png":
            "**Fig. 6 — E5/E6 dose-response at the prespecified layer over α ∈ {0.5, 1, 2, 4, 8}.** "
            "Both |E5| and |E6| grow monotonically with α; the specificity ratio |E5|/|E6| is "
            "non-monotonic and bottoms out near the prespecified α=4.0.",
        "fig7_geometry_svd.png":
            "**Fig. 7 — E8 SVD spectrum of the 12-subtype contrastive-direction matrix.** One "
            "dominant singular value, cliff to a near-uniform tail; participation ratio ≈ 2.5. "
            "Label-shuffle null (§4.10 M2) rules out a noise-floor explanation.",
        "fig8_geometry_cosine.png":
            "**Fig. 8 — E9 pairwise cosine among the 12 per-subtype directions.** All positive, "
            "mean ≈ 0.59, supporting a shared 'validate-this-frame' direction across distortion "
            "subtypes.",
        "fig9_geometry_subspace.png":
            "**Fig. 9 — E8/E9 per-layer geometry summary.** Left: participation ratio by layer "
            "(stable in interior layers, spikes at L0 which is style-confounded). Right: "
            "per-subtype top-5 subspace variance retained.",
    }
    cells.append(code(
        "from IPython.display import Image, display, Markdown\n"
        "import os\n"
        f"captions = {captions!r}\n"
        "for f in sorted(os.listdir('figures')):\n"
        "    if f.endswith('.png'):\n"
        "        cap = captions.get(f, f'**{f}**')\n"
        "        display(Markdown(cap))\n"
        "        display(Image(f'figures/{f}'))\n"
    ))

    # Mitigation experiments (M1 GroupKFold probe, M2 label-shuffle null, M3 held-out Cohen's d)
    cells.append(md("## 6b. Mitigation experiments (M1–M3)\n\n"
                    "These three supplementary analyses directly test Limitations 7 (pairing-leakage), "
                    "11 (noise-floor), and 14 (in-sample Cohen's d) from the paper. They are cheap "
                    "to run (~5 min on A100) and save to `results/mitigations.json`. See "
                    "`paper/paper.md §4.10` for interpretation."))
    mitig_src = (ROOT / "mitigation_experiments.py").read_text()
    mitig_cut = mitig_src.rfind("if __name__ ==")
    if mitig_cut > 0:
        mitig_src = mitig_src[:mitig_cut].rstrip() + "\n"
    mitig_src = mitig_src.replace("import reference  # noqa: E402\n", "")
    mitig_src = re.sub(r"\breference\.(?!py\b)", "", mitig_src)
    mitig_src = _patch_root(mitig_src)
    cells.append(code(mitig_src))
    cells.append(code(
        "import argparse\n"
        "mitig_args = argparse.Namespace(model='7b', device=None,\n"
        "                                n_layers=16, n_per_cat=0, n_shuffles=100)\n"
        "main(mitig_args)\n"
    ))
    cells.append(code(
        "import json\n"
        "mit = json.load(open('results/mitigations.json'))\n"
        "print('=== M1 GroupKFold vs StratifiedKFold ===')\n"
        "for l in mit['config']['sampled_layers']:\n"
        "    gk = mit['M1_group_kfold_probe']['group_kfold'][str(l)]['auc_mean']\n"
        "    sk = mit['M1_group_kfold_probe']['stratified_kfold_baseline'][str(l)]['auc_mean']\n"
        "    print(f'  L{l}: GroupKFold AUC = {gk:.4f}   StratifiedKFold AUC = {sk:.4f}   Δ = {gk-sk:+.4f}')\n"
        "print('\\n=== M2 Label-shuffle null (geometry at target layer) ===')\n"
        "for k, v in mit['M2_geometry_shuffle_null'].items():\n"
        "    print(f'  {k}: {v}')\n"
        "print('\\n=== M3 Held-out Cohen\\'s d ===')\n"
        "for l in mit['config']['sampled_layers']:\n"
        "    v = mit['M3_heldout_cohens_d'][str(l)]\n"
        "    print(f\"  L{l}: in-sample={v['cohens_d_in_sample']:.2f}  held-out={v['cohens_d_heldout']:.2f}  optimism={v['optimism_bias']:+.2f}\")\n"
    ))

    # Behavioural side-by-side demo (inline copy of behavioral_demo.py)
    cells.append(md("## 7. Behavioural demo (qualitative side-by-side)\n\n"
                    "Greedy generations with and without the projection-ablation / negative-steering "
                    "intervention at the pre-registered layer, for one stimulus per cognitive-distortion "
                    "subtype. The quantitative log-prob shifts in §4 are the headline; this is the "
                    "qualitative companion."))
    behav_src = (ROOT / "behavioral_demo.py").read_text()
    behav_main = behav_src[:behav_src.rfind('if __name__ ==')].rstrip() + "\n"
    # In the notebook, reference.py functions live in the global namespace
    # rather than under a `reference` module — strip the prefix and the import.
    behav_main = behav_main.replace("import reference  # share the helpers\n", "")
    # word-boundary replace: don't mangle "reference.py" in docstrings
    behav_main = re.sub(r"\breference\.(?!py\b)", "", behav_main)
    behav_main = _patch_root(behav_main)
    cells.append(code(behav_main))
    cells.append(code(
        "import argparse\n"
        "behav_args = argparse.Namespace(model='7b', device=None, n_demo=12,\n"
        "                                n_layers=16, max_new_tokens=120, alpha=4.0)\n"
        "main(behav_args)\n"
    ))
    cells.append(code(
        "print(open('results/behavioral_demo.md').read())\n"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
