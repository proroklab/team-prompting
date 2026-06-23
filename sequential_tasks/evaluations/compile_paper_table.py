"""Compile a paper-ready per-variant success-rate table across all methods.

Reads the JSON outputs of:
  * per_variant_eval.py — for the four RL-trained methods (ours-RNN, tiny-RNN, SLM, LATMOS)
  * evaluations/baselines/online_llm_planner/evaluate.py — for the LLM planner baseline

Both formats expose the per-variant aggregate under the top-level "aggregate"
key with the same field names ("success_rate", "mean_episode_length" /
"mean_steps", etc.), so unifying them is just dispatching on which name is
present.

Usage (defaults to the canonical output paths used elsewhere in this repo):
    PYTHONPATH=. python -m evaluations.compile_paper_table

Override the input paths or add a method:
    --rnn evaluations/per_variant_rnn.json
    --tiny-rnn evaluations/per_variant_tiny_rnn.json
    --slm evaluations/per_variant_slm.json
    --latmos evaluations/per_variant_latmos.json
    --llm-planner evaluations/baselines/online_llm_planner/results.json
    --output evaluations/paper_table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VARIANTS = ["free_order", "constrained_a_first", "constrained_b_first", "heavy_strict_ab"]
VARIANT_LABELS = {
    "free_order": "free",
    "constrained_a_first": "A-first",
    "constrained_b_first": "B-first",
    "heavy_strict_ab": "heavy",
}


def _load_aggregate(path: Path) -> Optional[Dict]:
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("aggregate", data)


def _extract_variant_row(agg: Optional[Dict]) -> Dict[str, Optional[float]]:
    """Return a uniform dict {variant_name: success_rate}. None where missing."""
    out = {v: None for v in VARIANTS + ["__overall__"]}
    if agg is None:
        return out
    for v in VARIANTS + ["__overall__"]:
        d = agg.get(v, {})
        sr = d.get("success_rate")
        if sr is not None:
            out[v] = float(sr)
    return out


def _format_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def main() -> int:
    p = argparse.ArgumentParser()
    # "Ours (RNN)" is the 0.16M Tiny variant. The earlier 5M run is no longer
    # in the headline table — it's kept on disk as supplementary data but is
    # not part of the paper claim.
    p.add_argument("--rnn", default="evaluations/per_variant_tiny_rnn.json", type=Path)
    p.add_argument("--slm", default="evaluations/per_variant_slm.json", type=Path)
    p.add_argument("--latmos", default="evaluations/per_variant_latmos.json", type=Path)
    p.add_argument("--llm-planner", default="evaluations/baselines/online_llm_planner/results.json", type=Path)
    p.add_argument("--output", default=Path("evaluations/paper_table.md"), type=Path)
    args = p.parse_args()

    methods: List[Tuple[str, str, Path]] = [
        ("Ours (RNN)",              "0.16M",   args.rnn),
        ("Distilled SLM",           "26.8M",   args.slm),
        ("LATMOS",                  "1.5M",    args.latmos),
        ("LLM planner (Llama 3.3 70B, online)", "n/a (70B)", args.llm_planner),
    ]

    # Build the markdown table.
    header = "| Method | Encoder | " + " | ".join(VARIANT_LABELS[v] for v in VARIANTS) + " | Overall |"
    sep = "|---|---|" + "|".join(["---"] * (len(VARIANTS) + 1)) + "|"
    rows: List[str] = [header, sep]

    print(header)
    print(sep)
    for label, params_label, path in methods:
        agg = _load_aggregate(path)
        sr = _extract_variant_row(agg)
        if all(v is None for v in sr.values()):
            cells = [f"_{label}_", params_label] + ["—"] * (len(VARIANTS) + 1)
            row = "| " + " | ".join(cells) + " |"
            rows.append(row)
            print(row + "   (no data — missing JSON at " + str(path) + ")")
            continue
        cells = [label, params_label] + [_format_pct(sr[v]) for v in VARIANTS] + [_format_pct(sr["__overall__"])]
        row = "| " + " | ".join(cells) + " |"
        rows.append(row)
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nWrote → {args.output}")

    # Quick diagnostic: per-variant comparison of the methods (max gap)
    print("\n=== variant gap diagnostic ===")
    print(f"{'variant':<24s}{'min':>8s}{'max':>8s}{'spread':>10s}")
    for v in VARIANTS:
        srs = []
        for _, _, path in methods:
            agg = _load_aggregate(path)
            sr = _extract_variant_row(agg).get(v)
            if sr is not None:
                srs.append(sr)
        if not srs:
            continue
        lo, hi = min(srs), max(srs)
        print(f"{VARIANT_LABELS[v]:<24s}{lo*100:>7.0f}%{hi*100:>7.0f}%{(hi-lo)*100:>9.0f} pp")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
