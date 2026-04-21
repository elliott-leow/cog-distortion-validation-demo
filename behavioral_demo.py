"""Side-by-side behavioural demo of the projection-ablation intervention.

Standalone: re-extracts the distortion-acceptance direction at the
intervention layer from the activation set, then generates greedy
continuations for a small held-out set of distortion prompts both with
and without the intervention. Saves transcripts to results/behavioral_demo.json
and a markdown table to results/behavioral_demo.md.

Designed to be run AFTER reference.py to demonstrate qualitatively what the
intervention does to free-form generation; the quantitative log-prob signal
in reference.py establishes the statistical claim.

Usage:
    python behavioral_demo.py --model 1b --device mps
    python behavioral_demo.py --model 7b --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

import reference  # share the helpers

ROOT = Path(__file__).parent
STIM_DIR = ROOT / "stimuli"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main(args):
    reference.set_seeds()
    device = reference.select_device(args.device)
    model_id = reference.MODEL_IDS[args.model]
    print(f"Loading {model_id} on {device}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = reference.pick_layers(n_layers,
                                   n_keep=args.n_layers if args.n_layers > 0 else None)
    intervention_layer = layers[len(layers) // 2]
    print(f"Intervention layer L{intervention_layer} (median sample of {len(layers)})")

    # Re-extract the distortion direction at the intervention layer.
    raw_dist = json.load(open(STIM_DIR / "cognitive_distortions.json"))
    extract_stim = reference.stratified_sample(raw_dist, n_per_subcat=None)
    print(f"Extracting on {len(extract_stim)} distortion stimuli ...")
    syc_acts, ther_acts = reference.extract_paired(
        model, tokenizer, extract_stim,
        "sycophantic_completion", "therapeutic_completion",
        [intervention_layer], desc="extract",
    )
    d_dist = reference.contrastive_direction(syc_acts, ther_acts)[intervention_layer]
    print(f"Direction extracted; norm={d_dist.norm():.3f}")

    # Pick demo stimuli (one per subcategory, capped at args.n_demo).
    by_cat = {}
    for s in raw_dist:
        by_cat.setdefault(s["subcategory"], []).append(s)
    demo_stim = [sorted(by_cat[c], key=lambda x: x["id"])[0] for c in sorted(by_cat)]
    demo_stim = demo_stim[: args.n_demo]
    print(f"Generating side-by-side completions for {len(demo_stim)} demo stimuli ...")

    transcripts: List[dict] = []
    for s in demo_stim:
        prompt = reference.format_prompt(tokenizer, s["user_prompt"])
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Baseline
        with torch.no_grad():
            out_b = model.generate(
                ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(out_b[0][ids.shape[1]:], skip_special_tokens=True)
        # Negative steering at intervention_layer
        for intervention_name, hook_fn in [
            ("ablation", reference.projection_ablation_hook(d_dist)),
            ("negative_steering", reference.negative_steering_hook(d_dist, args.alpha)),
        ]:
            handle = model.model.layers[intervention_layer].register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    out_i = model.generate(
                        ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                handle.remove()
            if intervention_name == "ablation":
                ablation = tokenizer.decode(out_i[0][ids.shape[1]:], skip_special_tokens=True)
            else:
                steered = tokenizer.decode(out_i[0][ids.shape[1]:], skip_special_tokens=True)

        transcripts.append({
            "subcategory": s["subcategory"],
            "id": s["id"],
            "user_prompt": s["user_prompt"],
            "baseline": baseline,
            "ablation": ablation,
            "negative_steering_alpha": args.alpha,
            "negative_steering": steered,
        })
        print(f"  [{s['subcategory']}] done")

    out = {
        "model_id": model_id,
        "intervention_layer": intervention_layer,
        "alpha": args.alpha,
        "max_new_tokens": args.max_new_tokens,
        "transcripts": transcripts,
    }
    reference.save_json(out, RESULTS_DIR / "behavioral_demo.json")
    print(f"Saved {RESULTS_DIR / 'behavioral_demo.json'}")

    # Also save a markdown table for easy reading
    md = ["# Behavioural demo: side-by-side completions",
          "",
          f"- Model: `{model_id}`",
          f"- Intervention layer: L{intervention_layer}",
          f"- Negative-steering α: {args.alpha}",
          f"- Greedy decoding, max_new_tokens = {args.max_new_tokens}",
          ""]
    for t in transcripts:
        md.append(f"## {t['subcategory']} (id={t['id']})")
        md.append("")
        md.append(f"**User prompt:** {t['user_prompt']}")
        md.append("")
        md.append(f"**Baseline (no intervention):** {t['baseline']}")
        md.append("")
        md.append(f"**Projection-ablation:** {t['ablation']}")
        md.append("")
        md.append(f"**Negative steering (α={args.alpha}):** {t['negative_steering']}")
        md.append("")
        md.append("---")
        md.append("")
    (RESULTS_DIR / "behavioral_demo.md").write_text("\n".join(md))
    print(f"Saved {RESULTS_DIR / 'behavioral_demo.md'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(reference.MODEL_IDS), default="1b")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-demo", type=int, default=6,
                   help="number of demo stimuli (one per subcategory)")
    ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=4.0)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
