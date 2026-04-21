# Ralph Loop Iteration Plan

For each iteration, the orchestrator (this agent) will:
1. Dispatch the three reviewers below in parallel via three `Agent` tool calls in a single message.
2. Parse the three JSON blocks from their responses.
3. If any reviewer's `verdict` is `"has_issues"`, address the `must_fix` items first then `should_fix` items.
4. Save the iteration's reviewer outputs to `review/iteration_<N>/`.
5. Re-dispatch (back to step 1) until all three reviewers report `"zero_issues"`.

The reviewer prompts below are exact strings to pass as the `prompt` parameter to `Agent` — no orchestrator context leaks in.

---

## Reviewer 1 — methods (subagent_type: general-purpose, model: opus)

```
You are a strict, anonymous peer reviewer for a short paper on the mechanistic interpretability of cognitive-distortion validation in language models. You have NO context from any other conversation. Read only the artefacts I point you to and report issues. The paper, code, results, and figures live in `/Users/elliottleow/Projects/claude-clinical-sycophancy/`.

Read these files:
- `paper/paper.md`
- `results/results.json`
- `figures/*.png` (use Read on the PNGs to see them)
- `reference.py` (for Methods §3 you may need to verify what the code actually does)

You are evaluating the EXPERIMENTAL METHODOLOGY AND STATISTICS. Check, in order of priority:

1. Are the contrasts well-defined? Do the stimulus conditions make the claimed comparisons valid? Are there obvious confounds (length, lexical overlap, position, prompt format)?
2. Is the layer-selection procedure honest? Are best-layer claims pre-registered or post-hoc? Is multiple-testing accounted for?
3. Is the permutation / random-control null appropriate? Is `n` large enough? Are p-values and z-scores computed correctly given the null?
4. Does each numerical claim in `paper.md` match a value in `results.json`? Spot-check at least three numbers by grep+jq.
5. Does the SPECIFICITY definition match what the paper claims it tests? Are there missing controls (e.g., is the warmth direction itself probed for separability before being used as the off-target signal)?
6. Are the figures honestly summarising the data? Any chart that selectively reports a subset of layers / alphas without saying so?
7. Does the negative result on specificity get adequate treatment, or is it under-claimed / over-claimed?

You may run shell commands via Bash. Do NOT modify any files.

REVIEW PHILOSOPHY:
- Focus on substantive issues: things that would mislead a reader, invalidate a claim, or break reproducibility.
- Do not invent issues to look thorough.
- Distinguish must-fix (numbered claims unsupported, mathematical errors, missing controls, unhonest framing) from should-fix (clarity, secondary controls).
- If you have no issues, say so and set verdict to "zero_issues".

You MUST end your response with a single fenced JSON block of the exact schema:

```json
{
  "verdict": "zero_issues" | "has_issues",
  "must_fix": [
    {"location": "<file:line or section>", "issue": "<one sentence>", "why_it_matters": "<one sentence>"}
  ],
  "should_fix": [
    {"location": "...", "issue": "...", "why_it_matters": "..."}
  ]
}
```

Set verdict to "zero_issues" only when both must_fix and should_fix are empty.
```

---

## Reviewer 2 — code (subagent_type: general-purpose, model: opus)

```
You are a strict, anonymous code reviewer for the implementation behind a short interpretability paper. You have NO context from any other conversation. The project is at `/Users/elliottleow/Projects/claude-clinical-sycophancy/`.

Read these files:
- `reference.py`
- `notebook.ipynb` (the Colab version — embeds reference.py via a parser)
- `build_notebook.py` (the converter)
- `regenerate_figures.py`
- `fill_paper.py`
- `results/results.json` (to verify outputs match the code's claimed structure)

You are evaluating CODE CORRECTNESS AND REPRODUCIBILITY. Check, in order of priority:

1. Does `reference.py` actually compute what the docstrings and the paper claim? Walk the math: contrastive direction, projection-ablation hook, teacher-forced log-prob, Gram-Schmidt decomposition, SVD geometry.
2. Is the projection-ablation hook correctly applied to the residual stream? Does it persist for the full forward pass? Is the hook removed in a `finally` clause? Does it handle the (tensor, *aux) tuple return shape?
3. Are seeds set everywhere they should be? Is the same set of random direction vectors used across the layer sweep (so the random-control comparison is apples-to-apples)?
4. Are tensor dtypes / devices handled correctly when the model is bf16 on CUDA but the cached direction was computed on CPU as float32?
5. Are off-by-one errors plausible: chat-template prompt length vs `tokenizer.encode(formatted+completion)`, the `prompt_len-1` slice in `completion_logprob`, layer 0 vs layer 1?
6. Does the notebook actually run end-to-end on a fresh Colab instance? Are stimuli embedded correctly (round-trip valid)? Is the reference.py inclusion safe if the source moves?
7. Are exception paths sensible? Does `cross_domain_probe` handle the `roc_auc_score` ValueError when one class is absent? Does `within_domain_probe` handle small folds?
8. Are computed numbers internally consistent: does the per-layer cosine in `results.json` round-trip through `_at` correctly? Does `make_figures` use the right structure?
9. Is anything in `reference.py` dead code (e.g. `permutation_test_cosine` is defined but unused)? Should it be removed or wired in?

You may run shell commands via Bash, including `python reference.py --quick --device cpu` to verify the smoke test still passes. Do NOT modify any files.

REVIEW PHILOSOPHY:
- Focus on real bugs that would change reported numbers.
- Distinguish must-fix (correctness bugs, reproducibility breakers, security issues) from should-fix (style, dead code).
- If you have no issues, say so and set verdict to "zero_issues".

You MUST end your response with a single fenced JSON block of the exact schema:

```json
{
  "verdict": "zero_issues" | "has_issues",
  "must_fix": [
    {"location": "<file:line or function>", "issue": "<one sentence>", "why_it_matters": "<one sentence>"}
  ],
  "should_fix": [...]
}
```
```

---

## Reviewer 3 — paper (subagent_type: general-purpose, model: opus)

```
You are a strict, anonymous referee for a workshop on AI safety / interpretability. You have NO context from any other conversation. The paper is at `/Users/elliottleow/Projects/claude-clinical-sycophancy/paper/paper.md`. You may also read `results/results.json` and `figures/*.png` if you need to verify a claim.

You are evaluating the WRITTEN PAPER. Check, in order of priority:

1. Does the abstract make a claim that the body actually supports? Is there a single-sentence summary of the contribution?
2. Is the introduction motivated by a real problem? Does it cite or reference prior work appropriately? Is the gap to prior work clear?
3. Are all numbered references (e.g. `[N4]`, `[E5.shift]`) actually filled in with concrete values, or do placeholders remain?
4. Is the methods section sufficient to reproduce the experiments end-to-end without reading the code? Are the contrastive directions, the intervention hook semantics, the log-prob signal, and the specificity definition each unambiguous?
5. Are all claims in the results section consistent with the figures referenced? Is each cited number present in `results.json`?
6. Is the discussion honest about negative results? Does it overclaim? Does it under-claim?
7. Are the limitations and ethics sections substantive (not boilerplate)? Do they identify the actual failure modes of the work?
8. References: are the citations real, dated correctly, and used somewhere in the body? Is anything cited that isn't in the references?
9. Does the paper end with a clear conclusion or summary that a reader can take away?

Do NOT modify any files.

REVIEW PHILOSOPHY:
- Focus on issues that would harm a reader's understanding, mislead them, or hide important caveats.
- Distinguish must-fix (unsupported claims, broken refs, placeholders left, missing critical analysis) from should-fix (clarity, polish).
- If you have no issues, say so and set verdict to "zero_issues".

You MUST end your response with a single fenced JSON block of the exact schema:

```json
{
  "verdict": "zero_issues" | "has_issues",
  "must_fix": [
    {"location": "<file:line or section>", "issue": "<one sentence>", "why_it_matters": "<one sentence>"}
  ],
  "should_fix": [...]
}
```
```
