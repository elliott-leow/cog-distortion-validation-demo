# Mechanistic Interpretability of Cognitive-Distortion Validation

Self-contained code and notebook for the paper **Mechanistic Interpretability of Cognitive-Distortion Validation in Open-Weight LLMs** (Leow, 2026). Reproduces the full six-experiment pipeline (E1–E9) plus four mitigation analyses (M1–M4) on OLMo-family instruct models.

## Quick start — Google Colab (OLMo-3 7B)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elliott-leow/cog-distortion-validation-demo/blob/main/notebook.ipynb)

1. Open `notebook.ipynb` in Colab.
2. Runtime → Change runtime type → **A100 GPU** (40 GB).
3. Runtime → Run all.
4. End-to-end runtime ≈ 30 minutes for the main pipeline + ≈ 5 minutes for the mitigation block + ≈ 5 minutes for the behavioural demo.

The notebook is self-contained — it embeds the required stimuli inline, installs `transformers>=4.57`, and writes `results/results.json`, `results/mitigations.json`, `results/behavioral_demo.{json,md}` and nine PNG figures to the Colab filesystem. A commented cell under **§5** mounts Google Drive and copies the outputs to `MyDrive/clinical_sycophancy_7b_run`.

## Quick start — local 1B validation (Apple MPS or CPU)

Requires Python ≥ 3.10 and ≥ 8 GB RAM.

```bash
pip install "transformers>=4.57.0" accelerate torch numpy scipy scikit-learn matplotlib tqdm
python reference.py --device mps --n-per-cat 0 --n-intervene 20 --n-random 5 --n-layers 8
python mitigation_experiments.py --device mps --n-layers 8
python behavioral_demo.py --device mps
bash finalize_run.sh mps    # fills paper placeholders + regenerates figures
```

Replace `mps` with `cpu` on non-Mac platforms.

## What the pipeline does

| ID | Question | Output |
|----|----------|--------|
| E1 | Is the syc-vs-ther completion pair linearly separable in the residual stream? | Per-layer AUC + permutation test |
| E2 | Is `d_dist` separable from a warmth direction and a factual-sycophancy direction? | Gram-Schmidt decomposition (both orderings) |
| E3 | Where in the network does the feature localise? | Per-layer Cohen's d |
| E4 | Does the direction generalise across 12 distortion subtypes? | Leave-one-out probe AUC |
| E7–E9 | What is the geometry of the per-subtype direction set? | SVD spectrum, participation ratio, pairwise cosines |
| E5 | Does negative-steering along `d_dist` shift therapeutic preference? | Held-out-direction log-prob shift + paired bootstrap CI |
| E6 | Is that shift specific (does it leave warmth alone)? | Signed specificity ratio + random-direction null |
| M1 | Does probe saturation rely on stimulus-pairing leakage? | GroupKFold probe |
| M2 | Is the low-dimensional geometry a noise-floor artefact? | Label-shuffle null for PR + pairwise cos |
| M3 | How much of §4.3's Cohen's d is in-sample optimism? | Held-out Cohen's d |
| M4 | What does signed specificity say about safety alignment? | Signed-ratio reparameterisation |

## Repository layout

```
.
├── notebook.ipynb              # self-contained Colab notebook (OLMo-3 7B)
├── reference.py                # main pipeline (E1-E9) — local 1B validation
├── mitigation_experiments.py   # M1-M3 standalone pipeline
├── behavioral_demo.py          # qualitative side-by-side generation demo
├── build_notebook.py           # regenerate notebook.ipynb from the .py sources
├── fill_paper.py               # substitute results.json values into paper placeholders
├── regenerate_figures.py       # rebuild figures without re-running experiments
├── finalize_run.sh             # fill_paper + regenerate_figures + behavioral_demo + build_notebook
├── paper/paper.md              # the paper itself
├── stimuli/                    # cognitive-distortion + factual-control + v2 stimulus JSONs
├── results/                    # results.json, mitigations.json, behavioral_demo.{json,md}
├── figures/                    # 9 PNG figures
└── review/
    ├── REVIEWER_PROMPTS.md
    ├── literature_review_mitigations.md
    └── iteration_1/
```

## Model IDs

- 1B local: `allenai/OLMo-2-0425-1B-Instruct` (fp32 on MPS/CPU)
- 7B Colab: `allenai/Olmo-3-7B-Instruct` (bf16 on A100)

## Citation

```
Leow, E. (2026). Mechanistic Interpretability of Cognitive-Distortion
Validation in Open-Weight LLMs. Draft, this repository.
```

## License

MIT.
