"""LanguageUnit variant that uses LATMOS for the subtask embedding.

KEY DIFFERENCE from our method and from the SLM baseline: LATMOS does NOT
read the language summary at all — its only input is the observation/event
sequence. This is the headline contrast reviewers (R12) asked us to make
explicit. The "language" in LanguageUnitLATMOS is therefore just bookkeeping
(summary string is stored but ignored by the encoder).

State prediction: LATMOS only learns binary acceptance/rejection, NOT the
joint state. So we leave ``state_A`` / ``state_B`` / ``state_flat`` set by
``sample_sequence_dataset`` (which uses dataset ground truth at the chosen
position), and let the scenario's ``_step_world_state`` advance them via
observed events at runtime. This is faithful to LATMOS — its automaton
state is implicit in the latent, not explicitly decoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from scenarios.search_rescue.language import LanguageUnit
from sequence_models.search_rescue.latmos.model import LatmosModel

# Module-level singleton, set by load_latmos_model.
latmos_model: Optional[LatmosModel] = None


def load_latmos_model(
    model_path: str,
    device: torch.device | str,
) -> None:
    global latmos_model
    latmos_model = LatmosModel().to(device)
    if model_path and Path(model_path).exists():
        sd = torch.load(model_path, map_location=device)
        latmos_model.load_state_dict(sd)
    latmos_model.eval()


class LanguageUnitLATMOS(LanguageUnit):
    """Same state interface; subtask embedding sourced from LATMOS forward."""

    @torch.no_grad()
    def compute_subtask_embedding_rollout_from_rnn(self, env_index: torch.Tensor) -> None:
        if latmos_model is None:
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

        # LATMOS input: event sequence + lengths. NO language conditioning.
        events = self.event_sequence[idx_t]
        lengths = self.sequence_length[idx_t]
        # LATMOS expects lengths >= 1 for the GRU pack to work cleanly.
        lengths = lengths.clamp(min=1)

        h_proj, _y_logits = latmos_model.encode(events, lengths)

        b_range = torch.arange(idx_t.size(0), device=self.device)
        idx_in_seq = (lengths - 1).clamp(min=0)
        h_now = h_proj[b_range, idx_in_seq, :]

        self.subtask_embeddings[idx_t] = h_now
        # NOTE: deliberately do NOT update state_A / state_B / state_flat —
        # LATMOS has no state-classification head. Joint state stays as
        # set by sample_sequence_dataset (dataset ground truth at the sampled
        # position), and the scenario advances it via observed events.

    def compute_forward_rnn(self, event, y, h):
        raise NotImplementedError(
            "LanguageUnitLATMOS has no per-step forward — call "
            "compute_subtask_embedding_rollout_from_rnn after updating the "
            "event_sequence buffer."
        )
