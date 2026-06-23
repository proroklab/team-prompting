# sequential_tasks

The main code for **Prompting Robot Teams with Natural Language** — the full three-stage
pipeline and all simulated experiments (paper Sec. V-A/B/C and the simulated parts of V-D).

## Layout

| Directory | Stage | Role |
| --- | --- | --- |
| `sequence_models/` | 1 + 2 | Per-task LLM data generation, automaton walks, and the RNN distillation (`*/model_training/rnn_model.py`, `EventRNN` GRU + MLP decoder). |
| `scenarios/` | 1 + 3 | VMAS environment definitions and the `language.py` modules that map instructions → automata-state sequences (one subfolder per task, plus per-method variants for search_rescue: `*_rl`, `*_slm`, `*_tiny_rnn`, `*_latmos`). |
| `trainers/` | 3 | BenchMARL/MAPPO training entry points (`benchmarl_train.py`) and the GNN policy models (`models/`). |
| `evaluations/` | — | Paper benchmarks, ablations, robustness studies, table compilation, plus baselines (`baselines/online_llm_planner`). Pre-computed result JSONs and `paper_table*.md` are included. |
| `eval/` | — | Interactive single-episode rollout (`run_sequential_task.py`). |
| `conf/` | — | Hydra configs: `<task>/{algorithm,model,task,training,experiment,deployment}/`. |
| `sentences/` | 1 | LLM-generated natural-language instruction datasets per sub-task. |
| `utils/` | — | Logging and small helpers. |

## Tasks

- **four_rooms** — recurrence + language ablation (Sec. V-B, Fig. 4): *Ours* vs *Vanilla RL*
  vs *Tuned-Rewards RL*.
- **four_flags** — branching-DFA flag task; also used for zero-shot team-size scaling (6/9/12).
- **hit_the_switch** — sequential switch-pressing task.
- **search_rescue** — heterogeneous team (2 scouts + 2 carriers), parallel/branching automata
  (Sec. V-C, Tables I/II). See `scenarios/search_rescue/SPEC.md` for the task design.

## Running

See the top-level [README](../README.md#reproducing-the-experiments) for the end-to-end
Stage 1 → 2 → 3 commands and the evaluation scripts. Everything is Hydra-driven from `conf/`.

## Not included

Generated datasets, model checkpoints (`*.pth`), TensorBoard/W&B logs, and rendered
figures/videos are excluded (large and regenerable). Filesystem paths to these appear as
`/path/to/...` placeholders in scripts and configs — edit them for your environment.
