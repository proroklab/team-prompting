"""LanguageUnit variant that uses the distilled SLM instead of our EventRNN.

Drop-in replacement: same external interface as
``scenarios.search_rescue.language.LanguageUnit`` so the scenario only needs
to swap out the constructor (handled by ``search_rescue_scenario_slm.py``).

Internally, ``compute_subtask_embedding_rollout_from_rnn`` (kept under that
name to avoid touching the scenario) builds per-env text inputs from the
stored summary + event sequence prefix, tokenises in a single batch, and
runs one SLM forward pass. The pooled hidden goes through the trained
``h_head`` to become ``subtask_embeddings``; the ``state_head`` logits are
decoded into the joint state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from scenarios.search_rescue.language import (
    LanguageUnit, NUM_AUTOMATA_BITS,
)
from sequence_models.search_rescue.distilled_slm.data import format_input_text
from sequence_models.search_rescue.distilled_slm.slm_model import (
    DistilledSLM, load_tokenizer, DEFAULT_MODEL_NAME, N_LAYERS_KEEP,
)
from sequence_models.search_rescue.random_automaton_walk_search_rescue import (
    SUB_STATE_DIM,
)

# Module-level singletons, set by load_slm_model.
slm_model: Optional[DistilledSLM] = None
slm_tokenizer = None


def load_slm_model(
    model_path: str,
    device: torch.device | str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    n_layers_keep: int = N_LAYERS_KEEP,
) -> None:
    """Load the trained SLM weights and tokenizer; cache as module globals."""
    global slm_model, slm_tokenizer
    slm_model = DistilledSLM(model_name=model_name, n_layers_keep=n_layers_keep).to(device)
    if model_path and Path(model_path).exists():
        sd = torch.load(model_path, map_location=device)
        slm_model.load_state_dict(sd)
    slm_model.eval()
    slm_tokenizer = load_tokenizer(model_name)


class LanguageUnitSLM(LanguageUnit):
    """Same state + interface as LanguageUnit; forward pass uses SLM not RNN."""

    @torch.no_grad()
    def compute_subtask_embedding_rollout_from_rnn(self, env_index: torch.Tensor) -> None:
        """Override: build text inputs per env, batch-encode through SLM,
        decode joint state from state head, update subtask_embeddings.
        """
        if slm_model is None or slm_tokenizer is None:
            # Model not loaded yet (e.g. dataset_load=False at construction).
            return

        # Resolve env_index → list of int indices.
        if env_index is None:
            env_idx_list = list(range(self.batch_size))
        else:
            if torch.is_tensor(env_index):
                env_idx_list = env_index.flatten().tolist()
            elif isinstance(env_index, int):
                env_idx_list = [env_index]
            else:
                env_idx_list = list(env_index)

        # Build text inputs per env.
        texts: list[str] = []
        for idx in env_idx_list:
            length = int(self.sequence_length[idx])
            events_prefix = self.event_sequence[idx, :length].detach().cpu().tolist()
            summary = self.summary[idx] or ""
            texts.append(format_input_text(summary, events_prefix))

        # Tokenize and forward.
        enc = slm_tokenizer(
            texts,
            padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        h_pred, state_logits = slm_model.encode(enc["input_ids"], enc["attention_mask"])

        # Decode joint state from the [aut_bits | A_one_hot | B_one_hot] head.
        a_logits = state_logits[..., NUM_AUTOMATA_BITS : NUM_AUTOMATA_BITS + SUB_STATE_DIM]
        b_logits = state_logits[..., NUM_AUTOMATA_BITS + SUB_STATE_DIM :]
        state_A_idx = torch.argmax(a_logits, dim=-1)
        state_B_idx = torch.argmax(b_logits, dim=-1)
        flat = state_A_idx * SUB_STATE_DIM + state_B_idx

        idx_t = torch.tensor(env_idx_list, device=self.device)
        self.state_A[idx_t] = state_A_idx
        self.state_B[idx_t] = state_B_idx
        self.state_flat[idx_t] = flat
        self.subtask_embeddings[idx_t] = h_pred

    def compute_forward_rnn(self, event, y, h):
        """Stub — the SLM doesn't have a single-step forward in the same sense.

        For per-step deployment (rollout / hardware), call
        ``compute_subtask_embedding_rollout_from_rnn`` after appending the
        latest event to ``event_sequence``. This stub keeps the public API
        compatible if any downstream code still calls compute_forward_rnn.
        """
        raise NotImplementedError(
            "LanguageUnitSLM has no single-step RNN forward — call "
            "compute_subtask_embedding_rollout_from_rnn after updating the "
            "event_sequence buffer instead."
        )
