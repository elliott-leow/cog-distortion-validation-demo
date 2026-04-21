# Iteration 1 — Aggregated Issues

## Methods reviewer (M)

**Must-fix:**
- M1. §3.3 says "every layer for the 1B local run ([0, 1, ..., 15])" but pick_layers returned 9 sampled (`[0, 2, 4, 6, 8, 10, 12, 14, 15]`) because we passed `--n-layers 8`. Misreports the sampling grid; affects every per-layer claim.
- M2. Abstract/§4.9/§5.2 lead with "max specificity 49.20" but it's a post-hoc argmax with no multiple-testing correction, dominated by a near-zero denominator (E6 = +0.0026). Reads as evidence of a clean intervention; not.
- M3. n_random=5 random-direction null per (layer, intervention) is too small; every reported z-score (e.g. ablation z≈104) is computed against a 5-point distribution.
- M4. Inconsistent intervention `n`: §6 Limitation 2 says "30-50" but config.n_intervene=20, abstract reports n=20.
- M5. §4.4 LOO "lowest = all_or_nothing AUC=1.000, highest = should_statements AUC=1.000" is meaningless when all tied; combined with saturation, the claim does not discriminate "validate-frame" from "opening-word style".
- M6. §4.9 contains a malformed programmatic-fill artefact: "which \`that a specific intervention exists, but only at non-pre-registered layers\`".

**Should-fix:**
- AUC=1.000 saturation should be flagged next to the headline, not just in Limitation 7.
- Specificity ratio is absolute-valued; sign of E6 carries safety meaning.
- Direction extraction does not length-correct (mean-pooled), but log-prob signal does.
- Permutation test under saturation is uninformative.
- §3.1 calls 79/89/78 word counts "roughly length-matched" — therapeutic is ~12% longer.
- Fig 5 plots ablation and steering against random-null lines on the same y-axis but the nulls are not on the same scale.

## Code reviewer (C)

**Must-fix:**
- C1. `ROOT = Path(__file__).parent` at module top of `reference.py` and `behavioral_demo.py` raises NameError when the embedded source is executed inside a Colab notebook cell. Breaks self-contained notebook.

**Should-fix:**
- `build_notebook.py` does naive `replace("reference.", "")` which mangles "reference.py" in docstrings.
- No assertion that prompt-token-encoding is a strict prefix of prompt+completion-token-encoding.
- Stored `sweep[l]["ablation"]["alpha"] = 0.0` is misleading (ablation has no alpha).
- `make_figures` silently produces blank subplots if subspace data missing.
- `_completion_acts` docstring says "no truncation" but `completion_logprob` accepts truncation arg — minor inconsistency.

## Paper reviewer (P)

**Must-fix:**
- P1. §5 Discussion line 164 and §5.2 Conclusion line 190 say "OLMo-3 7B Instruct" but every cited number comes from OLMo-2 1B (per config). Misattributes claims.
- P2. §4.9 / §5.2 cite "49.196" as headline of achievable specificity without the denominator-near-zero caveat.
- P3. §4.9 sentence ungrammatical (same as M6).

**Should-fix:**
- §4.4 LOO tie text (same as M5).
- §4.6 / §4.8 backticked prose strings render awkwardly.
- References missing venue/year metadata: Stade et al. 2024, Lawrence et al. 2024, Leow 2026 (internal preprint).
- Abstract trailing backtick: "leave-one-out probing `1.000`" reads like template leakage.
- §3.2 vs §4 vs §5 inconsistent attribution of headline numbers to model.
- Section ordering §§4.5–4.9 jumps E4 → E7 → E8 → E9 → E5 → E6.
