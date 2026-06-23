# Prompting Robot Teams with Natural Language

Code release for the IEEE *Robotics and Automation Letters* (RA-L) paper:

> **Prompting Robot Teams with Natural Language**
> Eduardo Sebastián\*, Nicolas Pfitzer\*, Ajay Shankar, Amanda Prorok
> *IEEE Robotics and Automation Letters*, 2026. (\*equal contribution)
>
> 🌐 Project page & videos: https://sites.google.com/view/prompting-teams

---

## Overview

We present a framework that **prompts multi-robot teams with high-level natural-language
tasks**. The key idea is that a robot task is an *algorithm* that can be represented as a
**Deterministic Finite Automaton (DFA)**, and that a **Recurrent Neural Network (RNN)** can
encode many such automata. We *distill* the reasoning and sub-task decomposition produced by
a Large (Multi-modal) Language Model into a single tiny RNN, and align its internal states
with the semantics of each task. A **Graph Neural Network (GNN)** control policy, trained
with multi-agent RL, then interprets the RNN's latent state to drive decentralized,
real-time, onboard execution.

The pipeline has three stages:

1. **Automatic dataset generation** — prompt an LMM/LLM with task descriptions and ground the
   resulting sub-task decompositions in the target environment, producing a dataset `D` of
   automata walks and language embeddings.
2. **Distilling reasoning into an RNN** *(supervised)* — train a tiny GRU (`hidden = 1024`,
   projected `[512, 512, 64]` language embedding) that encodes the pool of task automata; a
   shallow MLP decoder aligns its latent state with the ground-truth DFA states.
3. **Decentralized collaboration with a GNN** *(reinforcement learning)* — train a
   language-conditioned multi-task GNN policy (MAPPO) that solves each sub-task using only
   local observations and neighbor messages.

---

## Repository structure

```
prompting-robot-teams/
├── sequential_tasks/            # Main pipeline + all simulated experiments (Sec. V-A,B,C,D)
│   ├── sequence_models/         # Stage 1+2: LLM data generation & RNN distillation (per task)
│   ├── scenarios/               # VMAS environments + language modules (per task)
│   ├── trainers/                # Stage 3: BenchMARL/MAPPO training + GNN policy models
│   ├── evaluations/             # Paper benchmarks, ablations, tables & robustness (Sec. V)
│   ├── eval/                    # Interactive single-episode rollout
│   ├── conf/                    # Hydra configs (task / model / algorithm / training)
│   ├── sentences/               # LLM-generated natural-language instruction datasets
│   └── utils/
├── language_driven_exploration/ # Single-stage language-driven exploration variant + ROS 2 node
│   ├── trainer/  scenarios/  evaluations/  sentence_embeddings/  conf/
│   └── deployment/              # ROS 2 deployment node (my_deployment.py)
├── robomaster_ws/               # Thin ROS 2 (colcon) workspace for the real-robot deployment
│   ├── setup_workspace.sh       # Links the package above + clones Freyja & robomaster_msgs
│   └── record_mission.sh
├── requirements.txt
├── LICENSE                      # MIT
└── CITATION.cff
```

The four tasks studied in the paper live under `sequential_tasks/{sequence_models,scenarios,conf}`:
**four_rooms** (Sec. V-B ablation), **four_flags**, **hit_the_switch**, and **search_rescue**
(Sec. V-C search-and-rescue with branching/parallel DFAs). The retrieve-the-flag deployment
(Sec. V-D) is run through `robomaster_ws` / `language_driven_exploration`.

---

## Installation

Python ≥ 3.9 (tested with 3.10). We recommend a fresh virtual environment.

```bash
git clone <this-repo> prompting-robot-teams
cd prompting-robot-teams
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchrl`, `tensordict`, `benchmarl`, `vmas`, `torch-geometric`,
`hydra-core`, `sentence-transformers`, `transformers`. The instruction encoder is
`thenlper/gte-large` (downloaded on first use).

> **API keys (optional).** Only Stage-1 data generation (Gemini) and the online-LLM-planner
> baseline (Groq / Llama 3.3 70B) need keys. Copy the template and fill in your own — the
> released pipeline and the RNN/GNN models do **not** require any key:
> ```bash
> cp sequential_tasks/sequence_models/api_keys.example.py sequential_tasks/sequence_models/api_keys.py
> # or: export GEMINI_API_KEY=...  GROQ_API_KEY=...
> ```
> `api_keys.py` is git-ignored.

---

## Reproducing the experiments

All runs use [Hydra](https://hydra.cc); configs live in `sequential_tasks/conf/`. Paths to
generated artifacts (datasets, checkpoints) appear as `/path/to/...` placeholders — edit them
to your local locations or pass them on the command line.

**Stage 1 — generate the reasoning dataset** (per task, e.g. `search_rescue`):
```bash
cd sequential_tasks
python sequence_models/search_rescue/random_automaton_walk_search_rescue.py   # automaton walks
python sequence_models/search_rescue/build_augmented_dataset_search_rescue.py # + LLM language
```

**Stage 2 — distill the RNN** that encodes the task automata:
```bash
python sequence_models/<task>/model_training/rnn_model.py
```

**Stage 3 — train the GNN MAPPO policy**:
```bash
python trainers/benchmarl_train.py    # selects task/model/algorithm from conf/
```

**Evaluation / paper figures & tables** (`sequential_tasks/evaluations/`):
```bash
python evaluations/benchmarl_evaluate_four_rooms.py   # Sec. V-B ablation (Fig. 4)
python evaluations/per_variant_eval.py                # Sec. V-C search-and-rescue (Tables I/II)
python evaluations/compile_paper_table.py             # aggregate per-variant JSON → Markdown tables
python evaluations/heldout_paraphrases.py             # paraphrase / event-drop robustness
```
Pre-computed result JSONs and the rendered Markdown tables (`paper_table*.md`) are included for
reference.

> **Note on artifacts.** Trained checkpoints (`*.pth`), generated datasets, TensorBoard/W&B logs
> and rendered figures/videos are **not** included — they are large (100+ GB) and fully
> regenerable from the scripts above. Reach out to the authors if you need the exact pretrained
> models behind the paper's numbers.

---

## Real-robot deployment (Sec. V-D)

The retrieve-the-flag deployment runs on DJI RoboMaster EP robots (ROS 2 Humble, onboard
Nvidia Jetson Orin) with motion-capture positioning, on top of the
[Freyja](https://github.com/unl-nimbus-lab/Freyja) flight-control stack.

```bash
cd robomaster_ws
./setup_workspace.sh    # links language_driven_exploration + clones Freyja & robomaster_msgs
colcon build --symlink-install --cmake-args -DNO_PIXHAWK=True
source install/setup.bash

# Launch the low-level controller, then the policy node:
ros2 launch src/Freyja/freyja_robomaster.launch.yaml tf_myframe:=robomaster_2
python src/language_driven_exploration/deployment/my_deployment.py
```

See `robomaster_ws/README.md` for details. Third-party packages (Freyja — GPLv3 — and
`ros2_robomaster_msgs`) are **not** redistributed here; `setup_workspace.sh` fetches them.

---

## Citation

```bibtex
@article{sebastian2026prompting,
  title   = {Prompting Robot Teams with Natural Language},
  author  = {Sebasti\'an, Eduardo and Pfitzer, Nicolas and Shankar, Ajay and Prorok, Amanda},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Note that the third-party Freyja stack used for deployment is
GPLv3 and is fetched separately (not covered by this license).

## Acknowledgments

This work was supported by ARL DCIST CRA W911NF-17-2-0181 and by a Leverhulme Trust Research
Project Grant. Built on [BenchMARL](https://github.com/facebookresearch/BenchMARL),
[TorchRL](https://github.com/pytorch/rl), [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
and [Freyja](https://github.com/unl-nimbus-lab/Freyja).
