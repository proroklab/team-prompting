"""LanguageUnit for Search-and-Rescue.

Differences from ``scenarios/four_flags/language.py``:

* The DFA is parallel — a *joint* state ``(s_A, s_B)`` with sub-states drawn
  from {unfound, found, loaded, delivered}. We expose the joint state as
  two tensors (``state_A``, ``state_B``) and a flat scalar id used by the
  scenario's goal-coord lookup.
* The label/state encoding follows the search_rescue random-walk format:
  ``[automaton_id_bits | A_one_hot | B_one_hot]`` (length STATE_DIM = 10).

Until the LLM-augmentation pass produces real ``y`` task embeddings,
``load_sequence_data`` accepts the raw random-walk JSON (events / states /
label only) and synthesises zero embeddings. The training pipeline will
overwrite these with real values once available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sequence_models.four_flags.model_training.rnn_model import EventRNN
from sequence_models.search_rescue.random_automaton_walk_search_rescue import (
    EVENT_DIM, STATE_DIM, SUB_STATES, SUB_STATE_INDEX, SUB_STATE_DIM, N_SUB,
    UNFOUND, FOUND, LOADED, DELIVERED,
)

MAX_SEQ_LEN = 20  # matches MAX_SEQ_STEPS in random_automaton_walk_search_rescue.py
NUM_AUTOMATA_BITS = STATE_DIM - N_SUB * SUB_STATE_DIM  # 10 - 8 = 2

# Flat indices for the joint state, mainly so other code can do
#   state_id = jointstate_to_flat[(s_A, s_B)]
JOINTSTATE_TO_FLAT: Dict[tuple, int] = {}
FLAT_TO_JOINTSTATE: Dict[int, tuple] = {}
_idx = 0
for sa in SUB_STATES:
    for sb in SUB_STATES:
        JOINTSTATE_TO_FLAT[(sa, sb)] = _idx
        FLAT_TO_JOINTSTATE[_idx] = (sa, sb)
        _idx += 1
NUM_FLAT_STATES = _idx  # 16 (4×4); reachable subset is 15 with mutex


def load_sequence_model(model_path, embedding_size, event_size, state_size, device):
    global sequence_model
    sequence_model = EventRNN(
        event_dim=event_size,
        y_dim=embedding_size,
        latent_dim=embedding_size,
        input_dim=64,
        state_dim=state_size,
        decoder=None,
        use_encoder=False,
    ).to(device)
    if model_path and Path(model_path).exists():
        sequence_model.load_state_dict(torch.load(model_path, map_location=device))
    sequence_model.eval()


class LanguageUnit:
    """State / event / embedding bookkeeping for a batched search-rescue env."""

    def __init__(self, batch_size: int, embedding_size: int, use_embedding_ratio: float, device: str = "cpu"):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.use_embedding_ratio = use_embedding_ratio
        self.device = device

        # Task-level
        self.task_embeddings = torch.zeros((batch_size, embedding_size), device=device)
        self.subtask_embeddings = torch.zeros((batch_size, embedding_size), device=device)
        self.event_sequence = torch.zeros((batch_size, MAX_SEQ_LEN, EVENT_DIM), device=device)
        self.sequence_length = torch.zeros((batch_size,), dtype=torch.int, device=device)

        # Joint state: per-env sub-state indices in [0..3]
        self.state_A = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.state_B = torch.zeros((batch_size,), dtype=torch.long, device=device)
        # Flat id for convenience (0..15)
        self.state_flat = torch.zeros((batch_size,), dtype=torch.long, device=device)
        # Which DFA variant is active in each env (0..3 → free, a_first, b_first, heavy).
        # Set per-env at sample time from the dataset's automaton_id; used by the
        # scenario's _detect_event_vec to gate transitions per variant so that
        # task language actually drives reward.
        self.automaton_id = torch.zeros((batch_size,), dtype=torch.long, device=device)

        self.summary = ["" for _ in range(batch_size)]
        self.response = ["" for _ in range(batch_size)]
        self.train_dict: Optional[dict] = None
        self.total_dict_size: int = 0

    # ------------------------------------------------------------------ #
    #  Dataset loading
    # ------------------------------------------------------------------ #

    def load_sequence_data(self, json_path: str, device: str = "cpu") -> None:
        """Load a JSON dataset of trace dicts.

        Schema follows ``record_trace`` in random_automaton_walk_search_rescue.py
        (events / states / label) and may also contain ``y``, ``h``, ``summary``,
        ``responses``, ``subtask_decoder_labels`` once the LLM-augmentation pass
        has run.
        """
        project_root = Path(__file__).resolve().parents[2]
        full = project_root / json_path
        with full.open("r") as f:
            data = json.load(f)

        if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
            np.random.shuffle(data)
        else:
            data = [data]

        out: dict = {}

        if all("states" in d for d in data):
            out["states"] = [d["states"] for d in data]

        if all("y" in d for d in data):
            out["task"] = torch.tensor([d["y"] for d in data], dtype=torch.float32, device=device)
        else:
            out["task"] = torch.zeros((len(data), self.embedding_size), dtype=torch.float32, device=device)

        if all("h" in d for d in data):
            out["subtasks"] = [torch.tensor(d["h"], dtype=torch.float32, device=device) for d in data]

        if all("events" in d for d in data):
            evs = []
            for d in data:
                e_all = torch.zeros((MAX_SEQ_LEN, EVENT_DIM), dtype=torch.float32, device=device)
                e = torch.tensor(d["events"], dtype=torch.float32, device=device)
                e_all[: e.shape[0], :] = e
                evs.append(e_all)
            out["event"] = torch.stack(evs)

        if all("summary" in d for d in data):
            out["summary"] = [d["summary"] for d in data]
        if all("responses" in d for d in data):
            out["responses"] = [d["responses"] for d in data]

        if all("label" in d for d in data):
            # First NUM_AUTOMATA_BITS of label[0] encode the automaton_id (MSB
            # first per ``_automaton_id_bits`` in random_automaton_walk_search_rescue).
            # For NUM_AUTOMATA=4, width=2, so id = bit0*2 + bit1.
            aut_ids = []
            for d in data:
                bits = d["label"][0][:NUM_AUTOMATA_BITS]
                aut_id = 0
                for b in bits:
                    aut_id = (aut_id << 1) | int(b)
                aut_ids.append(aut_id)
            out["aut_id"] = torch.tensor(aut_ids, dtype=torch.long, device=device)

        self.train_dict = out
        self.total_dict_size = len(data)

    def sample_sequence_dataset(self, env_index, forced_state: Optional[tuple] = None) -> None:
        """Pick a sample (and a position within it) for each requested env slot.

        ``forced_state`` lets the scenario request "give me a sample currently
        in joint state (s_A, s_B)" — useful for curriculum / debug. By default
        we pick a random sample and a random position.
        """
        if self.train_dict is None:
            return

        if env_index is None:
            env_index = torch.arange(self.batch_size, device=self.device)
        else:
            env_index = torch.atleast_1d(torch.tensor(env_index, device=self.device))

        n = env_index.shape[0]
        chosen_sample: list[int] = []
        chosen_pos: list[int] = []

        for _ in range(n):
            j = int(torch.randint(0, self.total_dict_size, (1,)).item())
            seq = self.train_dict["states"][j]
            # Position 0 = initial state. Pick anywhere in [0, len-1].
            pos = int(torch.randint(0, max(1, len(seq)), (1,)).item())
            if forced_state is not None:
                # try a few random samples to find one that has the desired joint state
                for _try in range(8):
                    seq = self.train_dict["states"][j]
                    matches = [k for k, s in enumerate(seq) if tuple(s) == forced_state]
                    if matches:
                        pos = int(np.random.choice(matches))
                        break
                    j = int(torch.randint(0, self.total_dict_size, (1,)).item())
            chosen_sample.append(j)
            chosen_pos.append(pos)

        # Write task embeddings + event sequence for each chosen env slot
        chosen_t = torch.tensor(chosen_sample, device=self.device)
        if "task" in self.train_dict:
            self.task_embeddings[env_index] = self.train_dict["task"][chosen_t]
        if "event" in self.train_dict:
            self.event_sequence[env_index] = self.train_dict["event"][chosen_t]
        if "aut_id" in self.train_dict:
            self.automaton_id[env_index] = self.train_dict["aut_id"][chosen_t]
        if "summary" in self.train_dict:
            for i, idx in enumerate(env_index):
                self.summary[int(idx)] = self.train_dict["summary"][chosen_sample[i]]
        if "responses" in self.train_dict:
            for i, idx in enumerate(env_index):
                pos_i = chosen_pos[i]
                resps = self.train_dict["responses"][chosen_sample[i]]
                pos_i = min(pos_i, len(resps) - 1) if resps else 0
                self.response[int(idx)] = resps[pos_i] if resps else ""

        # Fill in joint state from the sampled position
        for i, idx in enumerate(env_index):
            j, p = chosen_sample[i], chosen_pos[i]
            seq = self.train_dict["states"][j]
            p_eff = min(p, len(seq) - 1)
            sa, sb = seq[p_eff]
            self.state_A[int(idx)] = SUB_STATE_INDEX[sa]
            self.state_B[int(idx)] = SUB_STATE_INDEX[sb]
            self.state_flat[int(idx)] = JOINTSTATE_TO_FLAT[(sa, sb)]
            self.sequence_length[int(idx)] = max(1, p_eff)

        if "subtasks" in self.train_dict:
            self.compute_subtask_embedding_rollout_from_rnn(env_index)

    # ------------------------------------------------------------------ #
    #  Joint-state interface used by the scenario
    # ------------------------------------------------------------------ #

    def joint_state_str(self, env_idx: int) -> tuple:
        """Return ``(s_A_str, s_B_str)`` for env_idx, e.g. ('found', 'unfound')."""
        return SUB_STATES[int(self.state_A[env_idx])], SUB_STATES[int(self.state_B[env_idx])]

    def reset_all(self) -> None:
        self.task_embeddings.zero_()
        self.subtask_embeddings.zero_()
        self.event_sequence.zero_()
        self.sequence_length.zero_()
        self.state_A.zero_()
        self.state_B.zero_()
        self.state_flat.zero_()
        self.automaton_id.zero_()
        self.summary = ["" for _ in range(self.batch_size)]
        self.response = ["" for _ in range(self.batch_size)]

    def reset_env(self, env_index: int) -> None:
        self.task_embeddings[env_index].zero_()
        self.subtask_embeddings[env_index].zero_()
        self.event_sequence[env_index].zero_()
        self.sequence_length[env_index] = 0
        self.state_A[env_index] = 0
        self.state_B[env_index] = 0
        self.state_flat[env_index] = 0
        self.automaton_id[env_index] = 0
        self.summary[env_index] = ""
        self.response[env_index] = ""

    def observe_task_embeddings(self) -> torch.Tensor:
        return self.task_embeddings.flatten(start_dim=1, end_dim=-1) \
            if self.task_embeddings.dim() > 1 else self.task_embeddings

    def observe_subtask_embeddings(self) -> torch.Tensor:
        return self.subtask_embeddings.flatten(start_dim=1, end_dim=-1) \
            if self.subtask_embeddings.dim() > 1 else self.subtask_embeddings

    # ------------------------------------------------------------------ #
    #  RNN forward / rollout
    # ------------------------------------------------------------------ #

    def compute_subtask_embedding_rollout_from_rnn(self, env_index: torch.Tensor) -> None:
        """Roll the RNN forward over the loaded event sequence to current position."""
        e = self.event_sequence[env_index]
        y = self.task_embeddings[env_index].unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)
        lengths = self.sequence_length[env_index]

        mask = (
            torch.arange(e.size(1), device=lengths.device).unsqueeze(0).expand(lengths.size(0), -1)
            < lengths.unsqueeze(1)
        )

        state_logits, sequence = sequence_model._rollout(e, y, lengths)
        sequence = sequence * mask.unsqueeze(-1)

        # Decode joint state from the [aut_bits | A | B] head
        a_logits = state_logits[..., NUM_AUTOMATA_BITS : NUM_AUTOMATA_BITS + SUB_STATE_DIM]
        b_logits = state_logits[..., NUM_AUTOMATA_BITS + SUB_STATE_DIM :]
        state_A_idx = torch.argmax(a_logits, dim=-1)
        state_B_idx = torch.argmax(b_logits, dim=-1)
        idx_in_seq = (lengths - 1).clamp(min=0)
        b_range = torch.arange(env_index.size(0), device=env_index.device)
        sa_now = state_A_idx[b_range, idx_in_seq]
        sb_now = state_B_idx[b_range, idx_in_seq]

        self.state_A[env_index] = sa_now
        self.state_B[env_index] = sb_now
        self.state_flat[env_index] = sa_now * SUB_STATE_DIM + sb_now
        self.subtask_embeddings[env_index] = sequence[b_range, idx_in_seq, :]

    def compute_forward_rnn(self, event: torch.Tensor, y: torch.Tensor, h: torch.Tensor):
        """Single forward step of the RNN — used at deployment / rollout time."""
        return sequence_model._forward(e=event, y=y, h=h)
