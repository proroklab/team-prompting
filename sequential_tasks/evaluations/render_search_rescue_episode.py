"""Render a single search-rescue episode using a trained policy checkpoint.

Bypasses BenchMARL's eval loop (which doesn't always save videos in restore
mode) and uses the env's native rgb_array render to dump frames as MP4.

Usage:
    PYTHONPATH=. python -m evaluations.render_search_rescue_episode \
        --checkpoint trainers/experiments/.../checkpoints/checkpoint_12000000.pt \
        --output evaluations/eval_render.mp4 \
        --max-steps 800 \
        --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import torch
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", default=Path("evaluations/eval_render.mp4"), type=Path)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--config-name", default="config_search_rescue_rl")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name=args.config_name)
    print(f"Loaded config '{args.config_name}'")

    experiment_name = list(cfg.keys())[0]
    cfg[experiment_name].experiment.restore_file = str(args.checkpoint.resolve())
    cfg[experiment_name].experiment.render = True
    cfg[experiment_name].experiment.evaluation_episodes = 1
    sub = cfg[experiment_name]

    print(f"Restoring policy from: {args.checkpoint.name}")
    experiment = benchmarl_setup_experiment(sub, seed=args.seed, main_experiment=False)

    # Build a single env instance with batch_dim=1 for rendering.
    env = experiment.test_env

    # Roll one deterministic episode and capture frames.
    print(f"Rolling out 1 episode (max {args.max_steps} steps), capturing frames...")
    td = env.reset()

    frames = []
    success = False
    n_steps = 0
    for step in range(args.max_steps):
        with torch.no_grad():
            td = experiment.algorithm.get_policy_for_collection()(td)
        td = env.step(td)
        # The render call reads from env.scenario; output is RGB (H,W,3).
        try:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
        except Exception as exc:
            print(f"  render warning at step {step}: {exc}")

        td = td["next"]
        n_steps = step + 1
        if td.get(("done",), torch.tensor(False)).any():
            success = True
            break

    print(f"Episode finished after {n_steps} steps (early-terminated: {success})")
    print(f"Captured {len(frames)} frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(str(args.output), frames, fps=args.fps, codec="libx264")
        print(f"Wrote video → {args.output.resolve()}")
    else:
        print("No frames captured — render mode may not be wired in this env.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
