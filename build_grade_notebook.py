"""Build grade_notebook.ipynb for OLMo-3 7B Instruct DPO on Colab GPU.

Embeds reference.py + grade_reference.py as a single code cell (stripped of
the `from reference import ...` statement since both modules live in the
same notebook globals, and with `__main__` guards removed), plus inline
stimuli so no external downloads are needed.

Produces `grade_notebook.ipynb` at the repo root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
REF = ROOT / "reference.py"
GRADE = ROOT / "grade_reference.py"
STIM_DIR = ROOT / "stimuli"
OUT = ROOT / "grade_notebook.ipynb"


def md(text: str):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def _patch_root(src: str) -> str:
    return src.replace(
        "ROOT = Path(__file__).parent",
        "ROOT = Path(globals().get('__file__', '.')).resolve().parent if globals().get('__file__') else Path.cwd()",
    )


def _strip_main_guard(src: str) -> str:
    m = re.search(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""", src, re.MULTILINE)
    if m is None:
        raise RuntimeError("missing __main__ guard")
    return src[:m.start()].rstrip() + "\n"


def load_reference() -> str:
    return _patch_root(_strip_main_guard(REF.read_text()))


def load_grade() -> str:
    src = _patch_root(_strip_main_guard(GRADE.read_text()))
    # In the notebook both files are loaded into the same globals, so the
    # `from reference import ...` block is a re-import — harmless if it runs
    # after reference.py, but Python will complain if reference.py has been
    # inlined as source text (the module named `reference` may not exist).
    # We therefore strip that block.
    src = re.sub(
        r"^from reference import \((?:.|\n)*?\)\n",
        "# (reference imports resolved inline — both files merged in this cell)\n",
        src,
        count=1,
        flags=re.MULTILINE,
    )
    return src


def build():
    cells = []

    cells.append(md("""# GRADE for Cognitive-Distortion Validation — OLMo-3 7B Instruct DPO

Colab notebook running the four GRADE-derived experiments (G1, G3, G4, G5) on
**OLMo-3 7B Instruct DPO**. The headline question the notebook answers:

> *Does the model have the CAPACITY to produce therapeutic (CBT-reframing)
> continuations to distorted user inputs, or does it merely fail to default
> to them?*

See [`review/grade_findings.md`](review/grade_findings.md) in the repo for
methods and 1B numbers. This notebook reproduces the 7B numbers.

**Runtime:** ~15–25 min on A100 40GB, ~40 min on L4. Memory peak ~22 GB
(bf16 forward + backward). Safer on A100; L4 works with `--n-per-cat 2`.

**Based on:** Wang et al. (2026), *GRADE: Probing Knowledge Gaps in LLMs
through Gradient Subspace Dynamics*, arXiv:2604.02830."""))

    cells.append(md("## 1. GPU check & dependencies"))
    cells.append(code(
        "!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'no gpu'\n"
    ))
    cells.append(code(
        "# OLMo-3 needs transformers >= 4.57. Colab's defaults are usually older.\n"
        '%pip install -q "transformers>=4.57.0" accelerate\n'
    ))

    cells.append(md("## 2. Embed stimuli"))
    cells.append(code(
        "import os, json\n"
        "os.makedirs('stimuli', exist_ok=True)\n"
        "os.makedirs('results', exist_ok=True)\n"
        "os.makedirs('figures', exist_ok=True)\n"
        "os.makedirs('review', exist_ok=True)\n"
    ))
    for fname in ["cognitive_distortions.json"]:
        body = (STIM_DIR / fname).read_text()
        compact = json.dumps(json.loads(body))
        cells.append(code(
            f"# {fname}\n"
            f"open('stimuli/{fname}','w').write({compact!r})\n"
        ))
    # 100-item factual subset
    fact_full = json.loads((STIM_DIR / "v2_factual_control.json").read_text())
    fact_sub = sorted(fact_full, key=lambda x: x["id"])[:100]
    cells.append(code(
        "# v2_factual_control.json (100-item id-sorted subset, for matched factual baseline)\n"
        f"open('stimuli/v2_factual_control.json','w').write({json.dumps(fact_sub)!r})\n"
        "print('stimuli:', sorted(os.listdir('stimuli')))\n"
    ))

    cells.append(md("## 3. Inline both pipeline modules\n\n"
                    "`reference.py` provides the shared activation / hook / logprob utilities; "
                    "`grade_reference.py` adds the GRADE gradient + rank-ratio logic and the "
                    "four experiments. They are concatenated into a single cell so one run of "
                    "the cell populates the notebook namespace with every symbol."))
    cells.append(code(load_reference()))
    cells.append(code(load_grade()))

    cells.append(md("## 4. Configure and run\n\n"
                    "- `model='7b-dpo'` ⇒ `allenai/Olmo-3-7B-Instruct-DPO`.\n"
                    "- `n_per_cat=4` (48 distortion stim); raise to 0 (all 100) if you have "
                    "the VRAM budget.\n"
                    "- `n_layers=8` samples 8-of-~32 layers. Raise to 0 for all layers (slow).\n"
                    "- `alpha=4.0` negative-steering magnitude, matched to the host repo."))
    cells.append(code(
        "# Populate argparse defaults from parse_args so adding a new CLI\n"
        "# flag to grade_reference.py never silently breaks this cell with\n"
        "# AttributeError. Override only what's Colab-specific.\n"
        "grade_args = parse_args([\n"
        "    '--model', '7b-dpo',\n"
        "    '--n-per-cat', '4',\n"
        "    '--n-intervene', '20',\n"
        "    '--n-layers', '8',\n"
        "    '--alpha', '4.0',\n"
        "    '--n-random', '20',\n"
        "])\n"
        "print('grade_args:', vars(grade_args))\n"
        "run(grade_args)\n"
    ))

    cells.append(md("## 5. View results"))
    cells.append(code(
        "import json, os\n"
        "assert os.path.exists('results/grade_results.json')\n"
        "R = json.load(open('results/grade_results.json'))\n"
        "print('=== config ===')\n"
        "for k, v in R['config'].items(): print(f'  {k}: {v}')\n"
        "\n"
        "print('\\n=== G1 per-layer (T vs S rank_ratio_pos, paired) ===')\n"
        "for l, d in R['G1_rank_ratios_per_layer'].items():\n"
        "    p = d['pos']\n"
        "    t = p['paired_T_vs_S_dist']\n"
        "    print(f\"  L{l}: T={p['mean_T_dist']:.3f}  S={p['mean_S_dist']:.3f}  \"\n"
        "          f\"Δ={t['mean_diff']:+.3f}  t={t['t']:+.2f}  p_t={t['p_t']:.3f}  \"\n"
        "          f\"p_sign={t['p_sign']:.3f}  d_z={t['cohens_dz']:+.2f}  \"\n"
        "          f\"Holm={t.get('p_t_holm', float('nan')):.3f}\")\n"
        "\n"
        "print('\\n=== G5 capacity ===')\n"
        "g5 = R['G5_capacity_summary']\n"
        "for k in ['capacity_mean_therapeutic_dist','capacity_mean_sycophantic_dist','capacity_mean_therapeutic_factual']:\n"
        "    print(f'  {k}: {g5[k]:.4f}')\n"
        "for k in ['paired_T_vs_S_dist','welch_T_dist_vs_T_factual',\n"
        "          'cluster_bootstrap_T_vs_S_dist',\n"
        "          'two_sample_cluster_bootstrap_T_dist_vs_T_factual']:\n"
        "    print(f'  {k}: {g5[k]}')\n"
        "\n"
        "print('\\n=== G3 mechanism steering ===')\n"
        "g3 = R['G3_mechanism_steering']\n"
        "print(f\"  layer={g3['layer']}, alpha={g3['alpha']}\")\n"
        "print(f\"  cos(v*, d_baseline) = {g3['cos_vstar_d_baseline']:+.3f}\")\n"
        "print(f\"  v*         ΔE5={g3['v_star']['E5_shift']:+.3f}  ΔE6={g3['v_star']['E6_shift']:+.3f}  \"\n"
        "      f\"spec={g3['specificity_ratio_vstar']:.2f}\")\n"
        "print(f\"  d_baseline ΔE5={g3['d_baseline']['E5_shift']:+.3f}  ΔE6={g3['d_baseline']['E6_shift']:+.3f}  \"\n"
        "      f\"spec={g3['specificity_ratio_d_baseline']:.2f}\")\n"
        "\n"
        "print('\\n=== G4 consensus ===')\n"
        "g4 = R['G4_consensus_sharpening']\n"
        "print(f\"  subcats={len(g4['subcategories'])}  centred srank_pos={g4['stable_rank_pos']:.2f}\")\n"
    ))

    cells.append(md("## 6. Figures"))
    cells.append(code(
        "from IPython.display import Image, display, Markdown\n"
        "figs = [\n"
        "    ('grade_g1_rank_ratio_by_layer.png',\n"
        "     'G1: per-layer mean rank ratio. Lower = more activated capacity. '\n"
        "     'Therapeutic below sycophantic at deep layers = model has the therapeutic capacity.'),\n"
        "    ('grade_g5_capacity_hist.png',\n"
        "     'G5: per-stimulus capacity histogram. Blue = therapeutic on distortion, '\n"
        "     'red = sycophantic on distortion, green = therapeutic on factual control.'),\n"
        "    ('grade_g3_specificity.png',\n"
        "     'G3: specificity ratio |ΔE5|/|ΔE6|. v* (top-PC of residual-gradient contrast) '\n"
        "     'vs d_baseline (activation contrast from host repo).'),\n"
        "    ('grade_g4_consensus.png',\n"
        "     'G4: 12-subtype per-layer profile of (T − S) rank ratio. '\n"
        "     'Low centred stable rank ⇒ shared cross-layer capacity-gap signature.'),\n"
        "]\n"
        "for f, cap in figs:\n"
        "    display(Markdown(f'**{cap}**'))\n"
        "    display(Image(f'figures/{f}'))\n"
    ))

    cells.append(md("## 7. Persist to Drive (optional)"))
    cells.append(code(
        "# from google.colab import drive\n"
        "# drive.mount('/content/drive')\n"
        "# import shutil, os\n"
        "# out = '/content/drive/MyDrive/grade_olmo3_7b_dpo'\n"
        "# os.makedirs(out, exist_ok=True)\n"
        "# shutil.copy('results/grade_results.json', out)\n"
        "# shutil.copytree('figures', f'{out}/figures', dirs_exist_ok=True)\n"
        "# print('saved to', out)\n"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3",
                           "language": "python"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    size = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size:.1f} KB, {len(cells)} cells)")


if __name__ == "__main__":
    build()
