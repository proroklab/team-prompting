"""Per-variant evaluation: success rate / mean reward / mean episode length,
broken down by DFA variant (free_order, a_first, b_first, heavy_strict_ab).

Why this matters for the paper:
* Aggregate eval reward across all variants can't distinguish "policy that
  reads language and adapts" from "policy that learned one good average
  strategy". The per-variant breakdown disambiguates: a language-blind
  policy succeeds on whichever variant its strategy happens to match and
  fails on the others.
* This is the headline rebuttal table (R12 §2 — "what does language buy us
  over LATMOS?").

Usage:
    PYTHONPATH=. python -m evaluations.per_variant_eval \\
        --checkpoint trainers/experiments/.../checkpoint_12000000.pt \\
        --config-name config_search_rescue_rl \\
        --episodes-per-variant 20 \\
        --output evaluations/per_variant_rnn.json

Run once per method, then diff the JSONs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment


VARIANT_NAMES = ["free_order", "constrained_a_first", "constrained_b_first", "heavy_strict_ab"]


def _get_scenario_from_env(env):
    """Walk the env's transform stack down to the underlying VMAS scenario."""
    cur = env
    while True:
        # torchrl TransformedEnv has base_env
        if hasattr(cur, "base_env"):
            cur = cur.base_env
            continue
        # torchrl VmasEnv has _env (the actual vmas.Environment)
        if hasattr(cur, "_env"):
            return cur._env.scenario
        if hasattr(cur, "scenario"):
            return cur.scenario
        raise RuntimeError(f"Could not find scenario on env (type={type(cur)}); fix the traversal")


class VariantFilter:
    """Filter the LanguageUnit's loaded dataset to a single DFA variant.

    During eval rollouts the env resets repeatedly; sample_sequence_dataset
    picks a random trace from train_dict each time. We restrict train_dict
    so only traces with the target automaton_id are sampled, then restore
    when moving to the next variant.
    """

    def __init__(self, language_unit):
        self.lu = language_unit
        self._saved = None
        self._saved_total = None

    def apply(self, target_aut_id: int) -> int:
        """Return the number of traces matching the target variant."""
        if self.lu.train_dict is None:
            raise RuntimeError("LanguageUnit has no dataset loaded; can't filter.")
        if self._saved is None:
            self._saved = dict(self.lu.train_dict)
            self._saved_total = self.lu.total_dict_size

        aut_ids = self._saved["aut_id"]
        idx = torch.where(aut_ids == target_aut_id)[0]
        indices_list = idx.tolist()

        filtered: Dict = {}
        for k, v in self._saved.items():
            if isinstance(v, torch.Tensor):
                filtered[k] = v[idx]
            elif isinstance(v, list):
                filtered[k] = [v[i] for i in indices_list]
            else:
                filtered[k] = v
        self.lu.train_dict = filtered
        self.lu.total_dict_size = len(indices_list)
        return len(indices_list)

    def restore(self) -> None:
        if self._saved is not None:
            self.lu.train_dict = self._saved
            self.lu.total_dict_size = self._saved_total


def run_variant(env, policy, scenario, filt: VariantFilter, *, variant_id: int,
                episodes_per_variant: int, max_steps: int) -> List[Dict]:
    """Run N episodes on a single variant, return per-episode records."""
    n_traces = filt.apply(variant_id)
    print(f"  filter[{VARIANT_NAMES[variant_id]}]: {n_traces} matching traces in dataset")

    records: List[Dict] = []
    n_envs = env.batch_size[0] if len(env.batch_size) > 0 else 1
    rounds = (episodes_per_variant + n_envs - 1) // n_envs

    for round_i in range(rounds):
        td = env.reset()
        n_active_in_round = min(n_envs, episodes_per_variant - len(records))
        total_rewards = torch.zeros(n_envs, device=td.device)
        ep_lengths = torch.zeros(n_envs, dtype=torch.long, device=td.device)
        done_flags = torch.zeros(n_envs, dtype=torch.bool, device=td.device)

        for step in range(max_steps):
            with torch.no_grad():
                td = policy(td)
            td = env.step(td)

            # Reward: reduce to shape (n_envs,) by averaging across agents and
            # squeezing trailing singleton dims. The reward tensor varies in
            # shape: (n_envs,), (n_envs, n_agents), or (n_envs, n_agents, 1).
            try:
                rew = td["next", "agents", "reward"]
                while rew.dim() > 1:
                    if rew.shape[-1] == 1:
                        rew = rew.squeeze(-1)
                    else:
                        rew = rew.mean(dim=-1)
            except (KeyError, RuntimeError):
                rew = torch.zeros(n_envs, device=td.device)
            total_rewards += rew * (~done_flags).float()
            ep_lengths[~done_flags] = step + 1

            # Detect newly-done envs — reduce same way.
            try:
                new_d = td["next", "done"]
                while new_d.dim() > 1:
                    if new_d.shape[-1] == 1:
                        new_d = new_d.squeeze(-1)
                    else:
                        new_d = new_d.any(dim=-1)
            except (KeyError, RuntimeError):
                new_d = torch.zeros(n_envs, dtype=torch.bool, device=td.device)
            done_flags |= new_d.bool() & (~done_flags)

            if done_flags.all():
                break
            td = td["next"]

        for i in range(n_active_in_round):
            records.append({
                "variant": VARIANT_NAMES[variant_id],
                "success": bool(done_flags[i].item()),
                "episode_length": int(ep_lengths[i].item()),
                "total_reward_mean_over_agents": float(total_rewards[i].item()),
            })
            if len(records) >= episodes_per_variant:
                break
        if len(records) >= episodes_per_variant:
            break

    filt.restore()
    return records


def aggregate(records: List[Dict]) -> Dict:
    """Per-variant + overall aggregate stats."""
    by_variant: Dict[str, Dict] = {}
    for v in VARIANT_NAMES + ["__overall__"]:
        if v == "__overall__":
            group = records
        else:
            group = [r for r in records if r["variant"] == v]
        n = len(group)
        if n == 0:
            by_variant[v] = {"n_episodes": 0}
            continue
        n_success = sum(1 for r in group if r["success"])
        rewards = [r["total_reward_mean_over_agents"] for r in group]
        lengths = [r["episode_length"] for r in group]
        by_variant[v] = {
            "n_episodes": n,
            "success_rate": n_success / n,
            "mean_total_reward": sum(rewards) / n,
            "mean_episode_length": sum(lengths) / n,
            "median_episode_length": sorted(lengths)[n // 2],
        }
    return by_variant


def _apply_paraphrase_swap(scenario) -> None:
    """Patch the LanguageUnit so resets use held-out paraphrases.

    For each reset, the env first samples a (random) trace from the augmented
    dataset (we let this happen so the joint state initialisation is correct),
    then we OVERRIDE the task_embedding with the gte-large embedding of a
    held-out paraphrase matching the sampled variant. This tests whether the
    policy generalises to unseen but semantically-equivalent phrasings.
    """
    from sentence_transformers import SentenceTransformer
    from evaluations.heldout_paraphrases import HELDOUT_PARAPHRASES

    print("  loading gte-large to embed held-out paraphrases (CPU)...")
    enc = SentenceTransformer("thenlper/gte-large", device="cpu")

    embeddings_by_variant: Dict[int, torch.Tensor] = {}
    for v_id, v_name in enumerate(VARIANT_NAMES):
        paras = HELDOUT_PARAPHRASES[v_name]
        embs = enc.encode(paras, normalize_embeddings=False, convert_to_numpy=True)
        embeddings_by_variant[v_id] = torch.tensor(embs, dtype=torch.float32)
        print(f"    {v_name}: {len(paras)} held-out paraphrases embedded")

    lu = scenario.language_unit
    original = lu.sample_sequence_dataset

    def patched(env_index, forced_state=None):
        original(env_index, forced_state)
        # Override task_embedding with a random held-out one per env
        if env_index is None:
            idx_t = torch.arange(lu.batch_size, device=lu.device)
        elif torch.is_tensor(env_index):
            idx_t = env_index.flatten().to(lu.device)
        else:
            idx_t = torch.tensor([env_index] if isinstance(env_index, int) else list(env_index), device=lu.device)
        for i_local, i_env in enumerate(idx_t.tolist()):
            v_id = int(lu.automaton_id[i_env].item())
            pool = embeddings_by_variant[v_id]
            choice = torch.randint(0, pool.shape[0], (1,)).item()
            lu.task_embeddings[i_env] = pool[choice].to(lu.device)

    lu.sample_sequence_dataset = patched
    print("  paraphrase-swap patch installed")


def _apply_event_noise(scenario, p_drop: float) -> None:
    """Patch _detect_event_vec to drop true events with probability p_drop.

    R12 §5 explicitly asked about "noisy or delayed detection". Dropping true
    events (1→0) is the meaningful direction — it tests "policy keeps trying
    even when a real discovery/load/deliver is missed." Adding spurious events
    (0→1) would only inflate the success rate because the env's state machine
    would advance on fake bits, which isn't a real-world failure mode the
    reviewer cares about.

    CARRIERS_SYNCED is excluded because it's a coordination signal, not a
    progress event, and dropping it would just intermittently stop the
    stretcher physics — uncoupled from the detection-quality concern.
    """
    from sequence_models.search_rescue.random_automaton_walk_search_rescue import CARRIERS_SYNCED

    original = scenario._detect_event_vec

    def patched():
        e = original()
        # Drop each '1' with probability p_drop. Leaves 0s alone (no false +).
        drop_mask = (torch.rand_like(e) < p_drop) & (e > 0.5)
        drop_mask[:, CARRIERS_SYNCED] = False
        e = torch.where(drop_mask, torch.zeros_like(e), e)
        return e

    scenario._detect_event_vec = patched
    print(f"  event-noise patch installed (p_drop={p_drop:.2f}, drop-only mode)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config-name", default="config_search_rescue_rl")
    p.add_argument("--output", default=Path("evaluations/per_variant_results.json"), type=Path)
    p.add_argument("--episodes-per-variant", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--paraphrase-mode", choices=["none", "held-out"], default="none",
                   help="Replace task_embedding with held-out paraphrases (R17 §1, R9 §4)")
    p.add_argument("--event-noise", type=float, default=0.0,
                   help="Bit-flip probability for non-sync event bits (R12 §5)")
    args = p.parse_args()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name=args.config_name)
    experiment_name = list(cfg.keys())[0]
    cfg[experiment_name].experiment.restore_file = str(args.checkpoint.resolve())
    cfg[experiment_name].experiment.render = False
    cfg[experiment_name].experiment.evaluation_episodes = args.episodes_per_variant
    cfg[experiment_name].experiment.evaluation_deterministic_actions = True
    sub = cfg[experiment_name]

    print(f"Restoring policy from: {args.checkpoint.name}")
    experiment = benchmarl_setup_experiment(sub, seed=args.seed, main_experiment=False)
    env = experiment.test_env
    policy = experiment.algorithm.get_policy_for_collection()
    scenario = _get_scenario_from_env(env)
    print(f"Got scenario: {type(scenario).__name__}, batch_size={env.batch_size}")

    # Robustness patches (applied before any reset so the first sampled
    # episode in each variant already sees the perturbation).
    if args.paraphrase_mode == "held-out":
        _apply_paraphrase_swap(scenario)
    if args.event_noise > 0.0:
        _apply_event_noise(scenario, args.event_noise)

    filt = VariantFilter(scenario.language_unit)

    all_records: List[Dict] = []
    t_start = time.time()
    for variant_id, variant_name in enumerate(VARIANT_NAMES):
        print(f"\n=== variant {variant_id}: {variant_name} ===")
        records = run_variant(
            env, policy, scenario, filt,
            variant_id=variant_id,
            episodes_per_variant=args.episodes_per_variant,
            max_steps=args.max_steps,
        )
        all_records.extend(records)
        n_succ = sum(1 for r in records if r["success"])
        print(f"  {n_succ}/{len(records)} succeeded; "
              f"mean reward {sum(r['total_reward_mean_over_agents'] for r in records)/max(1,len(records)):.1f}; "
              f"mean len {sum(r['episode_length'] for r in records)/max(1,len(records)):.0f}")

    elapsed = time.time() - t_start
    agg = aggregate(all_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "config_name": args.config_name,
            "episodes_per_variant": args.episodes_per_variant,
            "max_steps": args.max_steps,
            "elapsed_s": round(elapsed, 1),
            "aggregate": agg,
            "episodes": all_records,
        }, f, indent=2)
    print(f"\nWrote results → {args.output}")

    # Compact paper-ready summary
    print("\n=== PER-VARIANT SUMMARY ===")
    print(f"{'variant':<25s}{'n':>5s}{'succ%':>8s}{'mean_R':>10s}{'mean_len':>10s}")
    for v in VARIANT_NAMES + ["__overall__"]:
        a = agg[v]
        if a["n_episodes"] == 0:
            continue
        print(f"{v:<25s}{a['n_episodes']:>5d}{a['success_rate']*100:>7.1f}%{a['mean_total_reward']:>10.1f}{a['mean_episode_length']:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
