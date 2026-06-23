# Search-and-Rescue Warehouse — Task Spec (v0.1, draft)

Status: **DRAFT — awaiting sign-off before implementation**
Owner: revision team
Purpose: address reviewer convergence on the 26-0977v1 paper that the existing benchmark tasks are too structured (linear DFAs, ≤5 states, no parallelism, weak coordination).

## 1. Why this task

It must, in one task, exercise everything reviewers said the existing benchmark misses:

| Reviewer concern | How this task addresses it |
|---|---|
| Linear DFAs only (R12, R13, R9, meta) | Branching at the dispatch step + a parallel region for cooperative transport |
| Weak multi-robot coordination (R17 §2) | Cooperative-transport sub-goal requires ≥2 carriers acting in synchrony |
| Tasks pre-conform to clean FSA abstraction (R17 §5) | Ambiguity in *which casualty first* and in *how to allocate carriers* — the prompt doesn't pin a unique automaton path |
| Limited language complexity (R9, R17 §1) | Prompt template intentionally varies role assignment, ordering constraints, and conditional clauses |
| Short horizon (R12 §1) | Median episode length ≥3× current tasks |

We also need this task to **stay implementable** within the existing VMAS / BenchMARL / RNN-distillation pipeline. The design below is deliberately the *minimal* thing that satisfies the reviewer asks, not the maximal warehouse.

## 2. World

- **Arena**: 2D continuous, walled rectangle. Recommended dims `x_semidim=4.0`, `y_semidim=3.0` (compatible with existing config style).
- **Static landmarks**:
  - 2 **medical bays** (placed at fixed corners — colour-coded, e.g. green and blue).
  - 2 **casualties** spawned at randomised positions in the interior (not in bays, separated by ≥ `min_casualty_sep`).
  - 1 **stretcher** spawned at a randomised interior position (not overlapping casualty/bay).
  - Optional: a few obstacles (walls / pillars) reusing the `Box` landmark pattern from `four_flags`.
- **Dynamic state**: each casualty has flags `discovered`, `picked_up`, `delivered_to_bay_id`. Stretcher has `carriers_docked` (count) and `carrying_casualty_id`.

We can reuse `LanguageGrid` + occupancy from `scenarios/exploration/` for the discover phase, and the landmark pattern from `four_flags` for casualties / stretcher / bays.

**Decision (D1):** *extend* `scenarios/exploration/` or *fork* into `scenarios/search_rescue/`?
Recommendation — **fork**, but copy/import freely from `exploration/`. Reasons: (a) `exploration/` has reward modules (`reward_exploration.py`, `reward_defend.py`, `reward_navigation.py`) that already encode incompatible reward signals, (b) keeping the new task isolated makes the ablation story cleaner — reviewers can see "new task, new code path" without worrying about regression on prior tasks. The existing `exploration/` keeps working untouched.

## 3. Agents (heterogeneous team)

Two roles, fixed at episode start, broadcast in the agent observation as a one-hot:

| Role | Count (default) | Capabilities | Restrictions |
|---|---|---|---|
| **Scout** | 2 | Lidar, fast (`v_range_scout` > carrier), can `discover` a casualty by entering its detection radius | Cannot dock with stretcher, cannot trigger `delivered` events |
| **Carrier** | 2 | Slower, can `dock` with stretcher (must be ≥2 docked simultaneously to lift), can move stretcher (and any loaded casualty) by collective drag | Cannot trigger `discovered` events on undiscovered casualties |

Default team size **N=4** (2+2). Same GNN/MAPPO architecture as in the paper; role one-hot enters the local observation.

For the **zero-shot scaling claim** the paper already makes (twelve agents in sim), we should also test {2+2, 3+3, 4+4, 6+6} at evaluation time without retraining.

## 4. Atomic propositions (event vector)

The event vector is what drives the DFA. Keep it small and decidable from world state:

| Index | Event | Trigger condition |
|---:|---|---|
| 0 | `casualty_A_found` | Any scout within `detect_radius` of casualty A while `discovered_A=False` |
| 1 | `casualty_B_found` | symmetric |
| 2 | `stretcher_loaded_A` | ≥2 carriers docked AND stretcher within `pickup_radius` of casualty A AND `picked_up_A=False` |
| 3 | `stretcher_loaded_B` | symmetric |
| 4 | `delivered_A` | stretcher (with casualty A loaded) within `bay_radius` of the assigned bay for A |
| 5 | `delivered_B` | symmetric |
| 6 | `carriers_synced` | exactly two carriers within docking distance of the stretcher in the same simulation step |

**`EVENT_DIM = 7`** (was 5 in `four_flags`).

`carriers_synced` is the new kind of event: it's a **multi-agent atomic proposition** that can't be evaluated from any single agent's local view. This is exactly what reviewers (R17 §2) said was missing. Detection is centralised (scenario-side), but the policy is still decentralised — agents only see their local observation + RNN state.

## 5. Automaton structure (the core novelty)

Two parallel sub-automata that synchronise at delivery, with a branching dispatch step. Concretely:

### Sub-automaton P_A (casualty A pipeline)
States: `unfound_A → found_A → loaded_A → delivered_A` (terminal)
Transitions:
- `unfound_A → found_A` on `casualty_A_found`
- `found_A → loaded_A` on `stretcher_loaded_A` AND `carriers_synced`
- `loaded_A → delivered_A` on `delivered_A`

### Sub-automaton P_B
Symmetric to P_A.

### Composition
- The two sub-automata run **in parallel** — at any time the joint state is `(s_A, s_B)`.
- **Mutex constraint**: only one of P_A, P_B can be in `loaded_*` at a time (only one stretcher). This is the **branching choice**: the team must decide the order. Modelled as a guard on the `loaded_*` transitions.
- **Final state**: both `delivered_A` and `delivered_B`.

So the DFA over the joint state has **15 reachable states** under the single-stretcher mutex (vs 3–5 in current tasks) including a true branching choice and a parallel region. This is the smallest design that gives both branching *and* parallelism without ballooning the state space.

### Language prompt → automaton variants
Three prompt families, each producing a different DFA variant (so the RNN must encode *all three* simultaneously, like it does for four-flags variants today):

1. **Free order, single carrier pair** — "Find the two casualties and bring each to a medical bay."
2. **Constrained order** — "Rescue [colour] casualty first." → adds an ordering guard.
3. **Heavy casualty** — "Casualty [X] needs both carriers because the stretcher is heavy." → degenerates to a strict sequence (no parallel region for that one). Acts as a sanity case.

Generating the offline LLM dataset = procedurally sample (template, parameters), call the offline LLM with the keyword-randomised prompt, parse the decomposition, validate against the ground-truth DFA, store. Reuse `seq_emb_builder_multi_one_dataset.py` machinery.

**Required extension to `random_automaton_walk.py`:** the `Automaton` class currently steps a single state; we need a `ParallelAutomaton` that holds a tuple of sub-states and applies all transitions per step, with cross-state guards. The `advance_case` machinery handles branching as-is.

## 6. Rewards

Three additive components (mirroring how `exploration/` splits its reward into modules):

- `r_discover` — sparse +R per `casualty_*_found` (scouts only).
- `r_transport` — shaped distance reward when a casualty is loaded and the stretcher closes on the right bay (carriers only).
- `r_sync` — small +bonus per step that `carriers_synced` is true while a casualty is in `found_*` state and the stretcher is en route to it (encourages cooperative formation).
- Terminal +R_term on full DFA acceptance.

Per-agent role-conditioned sparsification: scouts don't get transport reward; carriers don't get discovery reward. This forces the GNN communication to actually carry information across roles.

## 7. Success metrics (for the paper)

- **Task success rate** (full DFA acceptance within budget).
- **Sub-goal success rates** broken down by: discovery, dispatch correctness, cooperative pickup, delivery.
- **Coordination quality**: fraction of steps with `carriers_synced=True` while productive.
- **Generalisation**: zero-shot transfer to {3+3, 4+4, 6+6} agents.
- **Paraphrase robustness**: success on held-out paraphrasings (Part 3 of the action plan).

## 8. Code layout

Mirror the per-task pattern. New files:

```
SequentialTasks/
  scenarios/search_rescue/
    SPEC.md                                 (this file)
    __init__.py
    search_rescue_scenario.py              ← main VMAS scenario
    search_rescue_rl.py                    ← BenchMARL training entrypoint
    search_rescue_gnn_train.py             ← GNN policy trainer
    search_rescue_rollout.py               ← evaluation rollouts
    search_rescue_deployment.py            ← hardware deployment hook (later)
    language.py                            ← LanguageUnit instance, decoder loaders
    load_config.py                         ← Hydra config loader
  sequence_models/search_rescue/
    random_automaton_walk_search_rescue.py ← parallel-automaton extensions
    seq_emb_builder_multi_one_dataset.py   ← reuse, parameterise
    add_high_level_task_summary_search_rescue.py
    model_training/                        ← reuse rnn_model.py
    model_eval/
  conf/search_rescue/
    {algorithm,deployment,experiment,model,task,training}/   ← Hydra configs
```

Plus a one-line addition in `scenarios/__init__.py`.

## 9. Open questions for sign-off

1. **D1 (fork vs extend):** confirm we fork `scenarios/search_rescue/` rather than extend `exploration/`. Recommendation: fork.
2. **Team composition:** confirm 2 scouts + 2 carriers as the canonical setup. Alternative: 1 scout + 3 carriers (more carrier coordination, weaker discovery story).
3. **Number of casualties / bays:** 2+2 keeps the DFA at 9 states. 3+2 would push to ~16 reachable states — more impressive but training cost grows. Recommendation: ship 2+2 first, treat 3+2 as a stretch ablation.
4. **Heavy-casualty variant:** include in the *first* round of experiments, or hold as a follow-up if v0.1 trains cleanly?
5. **Hardware:** confirm the existing RoboMaster setup can support the role distinction (scouts vs carriers as different physical configurations, or just role-labelled in software?). This affects whether we can reuse the already-collected hardware data for the new task or need a fresh round.

Once these are resolved, Tasks #2 and #3 can start in parallel.
