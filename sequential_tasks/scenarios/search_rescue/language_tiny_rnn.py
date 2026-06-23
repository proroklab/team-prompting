"""LanguageUnit variant for the tiny RNN.

Differs from the base ``LanguageUnit`` (which assumes RNN hidden dim ==
embedding_size == 1024) in one place: ``subtask_embeddings`` is allocated
at the tiny RNN's actual latent_dim (default 128), so the RNN's hidden
state assigns into it cleanly.

The policy does not observe ``subtask_embeddings`` directly (only via the
joint state → goal_coords → rel_to_goal observation), so this shape change
is transparent to downstream code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from scenarios.search_rescue.language import LanguageUnit, NUM_AUTOMATA_BITS
from sequence_models.search_rescue.distilled_rnn_tiny.model import (
    TinyEventRNN, DEFAULT_LATENT_DIM, DEFAULT_INPUT_DIM,
)
from sequence_models.search_rescue.random_automaton_walk_search_rescue import SUB_STATE_DIM

# Module-level singleton (same pattern as SLM / LATMOS).
tiny_rnn_model: Optional[TinyEventRNN] = None


def load_tiny_rnn_model(
    model_path: str,
    device: torch.device | str,
    *,
    latent_dim: int = DEFAULT_LATENT_DIM,
    input_dim: int = DEFAULT_INPUT_DIM,
) -> None:
    global tiny_rnn_model
    tiny_rnn_model = TinyEventRNN(
        latent_dim=latent_dim,
        input_dim=input_dim,
    ).to(device)
    if model_path and Path(model_path).exists():
        sd = torch.load(model_path, map_location=device)
        tiny_rnn_model.load_state_dict(sd)
    tiny_rnn_model.eval()


class LanguageUnitTinyRNN(LanguageUnit):
    """Drop-in LanguageUnit using the tiny RNN.

    Same fields/methods as the base class, with ``subtask_embeddings``
    resized to the tiny RNN's latent_dim.
    """

    def __init__(
        self,
        batch_size: int,
        embedding_size: int,
        use_embedding_ratio: float,
        latent_dim: int = DEFAULT_LATENT_DIM,
        device: str = "cpu",
    ) -> None:
        super().__init__(batch_size, embedding_size, use_embedding_ratio, device)
        # Override subtask_embeddings to use the tiny RNN's latent_dim.
        # task_embeddings stays at embedding_size (= gte-large 1024-dim) because
        # the policy observes it as `sentence_embedding`.
        self.latent_dim = latent_dim
        self.subtask_embeddings = torch.zeros((batch_size, latent_dim), device=device)

    @torch.no_grad()
    def compute_subtask_embedding_rollout_from_rnn(self, env_index) -> None:
        if tiny_rnn_model is None:
            return

        if env_index is None:
            idx_t = torch.arange(self.batch_size, device=self.device)
        else:
            if torch.is_tensor(env_index):
                idx_t = env_index.flatten().to(self.device)
            elif isinstance(env_index, int):
                idx_t = torch.tensor([env_index], device=self.device)
            else:
                idx_t = torch.tensor(list(env_index), device=self.device)

        from scenarios.search_rescue.language import MAX_SEQ_LEN

        e = self.event_sequence[idx_t]
        y = self.task_embeddings[idx_t].unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)
        lengths = self.sequence_length[idx_t].clamp(min=1)

        state_logits, sequence = tiny_rnn_model._rollout(e, y, lengths)

        a_logits = state_logits[..., NUM_AUTOMATA_BITS : NUM_AUTOMATA_BITS + SUB_STATE_DIM]
        b_logits = state_logits[..., NUM_AUTOMATA_BITS + SUB_STATE_DIM :]
        state_A_idx = torch.argmax(a_logits, dim=-1)
        state_B_idx = torch.argmax(b_logits, dim=-1)

        b_range = torch.arange(idx_t.size(0), device=self.device)
        idx_in_seq = (lengths - 1).clamp(min=0)
        sa_now = state_A_idx[b_range, idx_in_seq]
        sb_now = state_B_idx[b_range, idx_in_seq]

        self.state_A[idx_t] = sa_now
        self.state_B[idx_t] = sb_now
        self.state_flat[idx_t] = sa_now * SUB_STATE_DIM + sb_now
        self.subtask_embeddings[idx_t] = sequence[b_range, idx_in_seq, :]

    def reset_all(self) -> None:
        super().reset_all()
        self.subtask_embeddings.zero_()  # ensure correct (latent_dim) shape preserved

    def reset_env(self, env_index: int) -> None:
        super().reset_env(env_index)
        self.subtask_embeddings[env_index].zero_()
