"""Tiny variant of EventRNN — same architecture, ~10× smaller (~160K params).

Param budget breakdown (latent_dim=128, input_dim=32):
  e_proj    (7→32)                  ~256
  y_proj    (1024→64→64→32 MLP)     ~72K
  GRUCell   (64, 128)               ~75K     ← dominant cost (3 * (i*h + h*h))
  state_head (192→64→10)            ~13K
  Total                             ~160K

Compared with the original 5M EventRNN (latent=1024, input=64) and the 1.5M
LATMOS baseline, this is the smallest model in the comparison — strongest
"explicit DFA distillation beats raw capacity" claim for the paper.

Drops the cosine-reconstruction head: the original EventRNN computed
`loss_recon_h` but **never added it to the optimizer loss** (only cls_loss
and decoder label_loss are minimised), so the reconstruction head was
purely dead weight. Removing it lets us cut the output dim without harm.
"""

from __future__ import annotations

from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Defaults sized to hit ~160K total params.
DEFAULT_LATENT_DIM = 128
DEFAULT_INPUT_DIM = 32
DEFAULT_Y_DIM = 1024       # gte-large embedding dim
DEFAULT_STATE_DIM = 10     # 2 aut bits + 4 + 4 sub-state bits


class TinyEventRNN(pl.LightningModule):
    """Pared-down EventRNN: e_proj + y_proj → GRUCell → state head.

    Keeps the exact same external interface as ``EventRNN._rollout`` so
    ``language_tiny_rnn.py`` and the rest of the pipeline can call it the
    same way.
    """

    def __init__(
        self,
        *,
        event_dim: int = 7,
        y_dim: int = DEFAULT_Y_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        input_dim: int = DEFAULT_INPUT_DIM,
        lr: float = 1e-4,
        cls_loss_weight: float = 8.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        D = latent_dim
        I = input_dim
        hidden_dim = max(D // 2, 4)

        # Event embedding (tiny — events are 7-dim 0/1 vectors)
        self.e_proj = nn.Linear(event_dim, I)

        # Task summary embedding compressor (1024 → I). The bulk of "task
        # information" lives in y; the y_proj's job is to compress it to
        # something the GRU can fuse with the event signal.
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, I),
        )

        # Recurrent core. Single GRUCell stepped manually for parity with
        # the original EventRNN (so the _rollout signature matches).
        self.rnn = nn.GRUCell(2 * I, D)

        # State classification head (only loss-bearing output).
        self.state_head = nn.Sequential(
            nn.Linear(D + 2 * I, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.lr = lr
        self.cls_w = cls_loss_weight

    # ------------------------------------------------------------------ #

    def _rollout(self, e: Tensor, y: Tensor, lengths: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """Same interface as ``EventRNN._rollout``.

        Returns (state_logits, sequence). ``sequence`` is the GRU hidden
        state per timestep — downstream code reads sequence[..., -1, :]
        as the subtask embedding.
        """
        B, T, _ = e.shape
        D = self.hparams.latent_dim

        fused = torch.cat([self.e_proj(e), self.y_proj(y)], dim=-1)  # (B, T, 2*I)

        h = e.new_zeros(B, D)
        preds = []
        for t in range(T):
            h = self.rnn(fused[:, t], h)
            preds.append(h.clone())
        out = torch.stack(preds, dim=1)  # (B, T, D)

        state_logits = self.state_head(torch.cat([out, fused], dim=-1))  # (B, T, state_dim)
        return state_logits, out

    # ------------------------------------------------------------------ #

    def _step(self, batch: dict) -> Tensor:
        e, y = batch["e"], batch["y"]
        labels = batch["label"]    # (B, T+1, state_dim)
        lengths = batch["lengths"]

        target = labels[:, 1:]     # post-initial state labels
        pred_cls, _ = self._rollout(e, y, lengths)

        mask = (
            torch.arange(target.size(1), device=lengths.device)
            .unsqueeze(0).expand(lengths.size(0), -1)
            < lengths.unsqueeze(1)
        )
        raw = F.binary_cross_entropy_with_logits(pred_cls, target, reduction="none").mean(dim=-1)
        loss = (raw * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log_dict({"train_loss_cls": loss}, prog_bar=True)
        return self.cls_w * loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log_dict({"val_loss_cls": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
