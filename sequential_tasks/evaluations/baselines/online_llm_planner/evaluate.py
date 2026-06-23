"""Online LLM-planner baseline — evaluation entry point.

Runs N episodes per DFA variant, picks task summaries from the same pool used
to build the augmented dataset, and writes per-episode + aggregate results.

Setup (one-time):
    export GROQ_API_KEY=gsk-...

Smoke test (cheap, ~$0.50–1, ~5 min wallclock):
    PYTHONPATH=. python -m evaluations.baselines.online_llm_planner.evaluate \
        --episodes-per-dfa 2 --max-steps 150 --output smoke_results.json

Full run (paper-grade, ~50 episodes per DFA × 4 DFAs = 200 episodes; cost
depends on how many episodes terminate early — rough estimate ~$15–30):
    PYTHONPATH=. python -m evaluations.baselines.online_llm_planner.evaluate \
        --episodes-per-dfa 50 --max-steps 300 --output llm_planner_results.json

Outputs (next to --output):
    <output>             — full per-episode records (long; contains every plan call)
    <output>.summary.csv — aggregate metrics: success rate, mean steps, mean latency, mean tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from scenarios.search_rescue.search_rescue_scenario_rl import SearchRescueScenario
from sequence_models.search_rescue.build_augmented_dataset_search_rescue import (
    SUMMARIES_BY_AUTOMATON,
)
from evaluations.baselines.online_llm_planner.planner import OnlineLLMPlanner, DEFAULT_MODEL
from evaluations.baselines.online_llm_planner.runner import run_one_episode, EpisodeRecord


def _build_scenario():
    """Construct a single batch_dim=1 scenario, mimicking the VMAS Environment wrap."""
    scenario = SearchRescueScenario()
    world = scenario.make_world(
        batch_dim=1,
        device=torch.device("cpu"),
        load_dataset=False,  # the LLM doesn't need the augmented dataset
    )
    scenario._world = world
    return scenario


def _aggregate(records: List[EpisodeRecord]) -> Dict[str, Dict[str, float]]:
    by_dfa: Dict[str, List[EpisodeRecord]] = defaultdict(list)
    for r in records:
        by_dfa[r.automaton_name].append(r)
    by_dfa["__overall__"] = records

    out = {}
    for name, group in by_dfa.items():
        n = len(group)
        n_success = sum(1 for r in group if r.success)
        steps = [r.n_steps for r in group]
        n_calls = [r.planner_stats["n_calls"] for r in group]
        latencies = [r.planner_stats["total_latency_s"] for r in group]
        in_toks = [r.planner_stats["total_input_tokens"] for r in group]
        out_toks = [r.planner_stats["total_output_tokens"] for r in group]
        cache_toks = [r.planner_stats["total_cache_read_tokens"] for r in group]
        out[name] = {
            "n_episodes": n,
            "success_rate": n_success / n if n else 0.0,
            "mean_steps": sum(steps) / n if n else 0.0,
            "mean_n_planner_calls": sum(n_calls) / n if n else 0.0,
            "mean_total_latency_s": sum(latencies) / n if n else 0.0,
            "mean_input_tokens_per_episode": sum(in_toks) / n if n else 0.0,
            "mean_output_tokens_per_episode": sum(out_toks) / n if n else 0.0,
            "mean_cache_read_tokens_per_episode": sum(cache_toks) / n if n else 0.0,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-per-dfa", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--replan-every", type=int, default=20)
    p.add_argument("--v-max", type=float, default=1.0)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--min-interval-s", type=float, default=0.0,
                   help="Min wallclock seconds between successive LLM calls. "
                        "Set to 6.5 for Flash free-tier (10 RPM).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--dfas", type=str, default="all",
        help="Comma-separated DFA names, or 'all'. Choices: free_order, "
             "constrained_a_first, constrained_b_first, heavy_strict_ab",
    )
    p.add_argument("--output", type=Path, default=Path("evaluations/baselines/online_llm_planner/results.json"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    planner = OnlineLLMPlanner(model=args.model, min_interval_s=args.min_interval_s)

    if args.dfas == "all":
        dfa_names = list(SUMMARIES_BY_AUTOMATON.keys())
    else:
        dfa_names = [d.strip() for d in args.dfas.split(",")]

    scenario = _build_scenario()
    print(f"Scenario constructed (batch_dim=1, agents={len(scenario.world.agents)}, model={args.model})")

    all_records: List[EpisodeRecord] = []
    t_start = time.time()
    for dfa in dfa_names:
        summaries = SUMMARIES_BY_AUTOMATON[dfa]
        print(f"\n=== {dfa} ({args.episodes_per_dfa} episodes) ===")
        for ep in range(args.episodes_per_dfa):
            summary = rng.choice(summaries)
            print(f"  ep {ep+1}/{args.episodes_per_dfa}: {summary[:80]}...")
            record = run_one_episode(
                scenario=scenario,
                planner=planner,
                summary=summary,
                automaton_name=dfa,
                max_steps=args.max_steps,
                replan_every=args.replan_every,
                v_max=args.v_max,
                verbose=args.verbose,
            )
            all_records.append(record)
            tag = "✓" if record.success else "✗"
            ps = record.planner_stats
            print(
                f"    {tag} steps={record.n_steps:>3d}  "
                f"planner_calls={ps['n_calls']:>2d}  "
                f"latency={ps['total_latency_s']:.1f}s  "
                f"in/out tokens={ps['total_input_tokens']}/{ps['total_output_tokens']}"
                + (f"  ({record.failure_reason})" if record.failure_reason else "")
            )

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed/60:.1f} min")

    aggregate = _aggregate(all_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        # Per-episode records are large (full plan_log per episode); keep them for inspection.
        json.dump({
            "model": args.model,
            "episodes_per_dfa": args.episodes_per_dfa,
            "max_steps": args.max_steps,
            "replan_every": args.replan_every,
            "elapsed_s": round(elapsed, 1),
            "aggregate": aggregate,
            "episodes": [asdict(r) for r in all_records],
        }, f, indent=2)
    print(f"Per-episode records → {args.output}")

    # Aggregate CSV (paper-friendly)
    summary_csv = args.output.with_suffix(".summary.csv")
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        cols = list(next(iter(aggregate.values())).keys())
        writer.writerow(["dfa"] + cols)
        for dfa, row in aggregate.items():
            writer.writerow([dfa] + [row[c] for c in cols])
    print(f"Aggregate CSV    → {summary_csv}")

    print("\n=== SUMMARY ===")
    overall = aggregate.get("__overall__", {})
    print(f"  success rate:           {overall.get('success_rate', 0):.2%}")
    print(f"  mean steps:             {overall.get('mean_steps', 0):.1f}")
    print(f"  mean planner calls/ep:  {overall.get('mean_n_planner_calls', 0):.1f}")
    print(f"  mean LLM latency/ep:    {overall.get('mean_total_latency_s', 0):.1f}s")
    print(f"  mean tokens/ep (in):    {overall.get('mean_input_tokens_per_episode', 0):.0f}")
    print(f"  mean tokens/ep (out):   {overall.get('mean_output_tokens_per_episode', 0):.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
