"""Re-generate figures from results/results.json without re-running the experiment.

Reconstructs the in-memory dicts that `make_figures` expects from the JSON
artefact. Useful when figure styling needs tweaking after a long run.
"""

from __future__ import annotations

import json
from pathlib import Path

import reference  # imports make_figures, FIGURES_DIR, etc.

ROOT = Path(__file__).parent
RES = ROOT / "results" / "results.json"


def _intify_keys(d):
    """JSON keys are strings; restore int keys for layer indices."""
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            ik = k
        out[ik] = _intify_keys(v) if isinstance(v, dict) else (
            [_intify_keys(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], dict) else v
        )
    return out


def main():
    res = json.loads(RES.read_text())
    cfg = res["config"]
    layers = cfg["sampled_layers"]
    target_layer = cfg["target_layer_pre_registered"]
    inter_layer = cfg["intervention_layer_pre_registered"]
    best_layer = cfg["best_descriptive_layer"]

    sweep = _intify_keys(res["E5_E6_intervention_sweep"]["layer_sweep"])
    alpha_sweep_raw = res["E5_E6_intervention_sweep"]["alpha_sweep_at_intervention_layer"]
    # alpha keys are floats (str in JSON) — convert
    alpha_sweep = {float(a): _intify_keys(v) for a, v in alpha_sweep_raw.items()}
    alpha_grid = sorted(alpha_sweep.keys())

    geometry_by_layer = _intify_keys(res.get("E7_E8_E9_geometry", {"by_layer": {}})["by_layer"]) or None

    reference.make_figures(res, layers, target_layer, best_layer, inter_layer,
                           sweep, alpha_sweep, alpha_grid, cfg["alpha_layer_sweep"],
                           geometry_by_layer)
    print(f"Regenerated figures in {reference.FIGURES_DIR}")


if __name__ == "__main__":
    main()
