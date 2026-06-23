"""Clean re-implementation of LATMOS (Zhan et al., IROS 2025, arxiv 2503.08090).

LATMOS = "Latent Automaton Task Model from Observation Sequences" — learns
a binary acceptance/rejection classifier over observation sequences, using
a recurrent sequence model whose latent state implicitly encodes a finite
automaton over task progress.

Key differences from our method (encoded explicitly so reviewers can verify):

  * Input: observation sequence only — NO language conditioning. This is the
    central thing reviewers (R12) asked us to compare against. Whatever
    automaton LATMOS learns must be inferable from observations alone.
  * Supervision: binary classification on (real trace) vs (synthetic negative
    = prefix + uniform-random-suffix). No state labels, no decompositions.
  * Output: h_k (latent state per step), used downstream as the task-state
    feature for the GNN policy — drop-in replacement for our subtask
    embedding.

Architecture choices match the paper's GRU variant (simpler than Attention,
recommended for our small event-vector inputs):
  * Task-specific observation encoder: 2-layer MLP (analog of their 1-layer
    CNN for image inputs; ours are 7-dim event vectors).
  * Sequence model: 1-layer GRU, hidden_dim=512 (their Ego-Exo4D Attention
    setting; we use the same for consistency across model classes).
  * Decoder: MLP with one hidden layer at avg(in,out) dim, LeakyReLU.
  * Final projection h → 1024 so it slots into our existing GNN policy
    (which expects ``subtask_embedding`` of size embedding_size=1024).
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Defaults match the LATMOS paper's GRU variant on Ego-Exo4D.
DEFAULT_FEAT_DIM = 64
DEFAULT_HIDDEN_DIM = 512
PROJ_OUT_DIM = 1024  # match our pipeline's embedding_size so GNN is identical


class LatmosObservationEncoder(nn.Module):
    """Task-specific encoder. Our 'observation' is the 7-dim event vector at
    each step; a small MLP is the symbolic analog of their 1-layer CNN.

    The paper also uses a *frozen* pre-trained encoder (XCLIP, Omnivore, etc.)
    in series for image/video inputs. There's no equivalent for our symbolic
    inputs, so we omit it — this is the cleanest faithful adaptation.
    """

    def __init__(self, event_dim: int = 7, feat_dim: int = DEFAULT_FEAT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(event_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, event_dim) → (B, T, feat_dim)
        return self.net(x)


class LatmosDecoder(nn.Module):
    """Per-paper: MLP w/ 1 hidden layer at (in+out)/2 dim, LeakyReLU.

    Output is a single logit per step — binary acceptance/rejection. We
    apply sigmoid externally during loss computation (BCEWithLogits).
    """

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        super().__init__()
        mid = max(1, (hidden_dim + 1) // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LeakyReLU(),
            nn.Linear(mid, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, T, hidden_dim) → (B, T)
        return self.net(h).squeeze(-1)


class LatmosModel(pl.LightningModule):
    """Full LATMOS pipeline: encoder → GRU → decoder, plus projection head
    so the latent state slots into our pipeline.
    """

    def __init__(
        self,
        *,
        event_dim: int = 7,
        feat_dim: int = DEFAULT_FEAT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        proj_out_dim: int = PROJ_OUT_DIM,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = LatmosObservationEncoder(event_dim, feat_dim)
        self.gru = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.decoder = LatmosDecoder(hidden_dim)
        # Project h_k → 1024 so our GNN sees the same shape as with the RNN.
        self.h_proj = nn.Linear(hidden_dim, proj_out_dim)
        self.lr = lr

    # ------------------------------------------------------------------ #

    def encode(self, event_seq: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """event_seq: (B, T, event_dim). Returns (h_proj (B,T,1024), y_logits (B,T))."""
        feats = self.encoder(event_seq)
        if lengths is None:
            h, _ = self.gru(feats)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_h, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True, total_length=event_seq.size(1))
        y_logits = self.decoder(h)
        h_proj = self.h_proj(h)
        return h_proj, y_logits

    # ------------------------------------------------------------------ #

    def _step(self, batch: dict) -> dict:
        # batch keys: "events" (B, T, event_dim), "label" (B, T), "mask" (B, T), "lengths" (B,)
        events = batch["events"]
        labels = batch["label"]
        mask = batch["mask"].float()

        _, y_logits = self.encode(events, batch["lengths"])
        bce = F.binary_cross_entropy_with_logits(y_logits, labels, reduction="none")
        loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)

        # Acceptance accuracy at the LAST valid step of each sequence.
        last_idx = (batch["lengths"] - 1).clamp(min=0)
        b_range = torch.arange(events.size(0), device=events.device)
        last_logit = y_logits[b_range, last_idx]
        last_label = labels[b_range, last_idx]
        pred = (last_logit > 0).float()
        acc = (pred == last_label).float().mean()

        return {"loss": loss, "acc_final": acc}

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log_dict({"train_loss": out["loss"], "train_acc_final": out["acc_final"]}, prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log_dict({"val_loss": out["loss"], "val_acc_final": out["acc_final"]}, prog_bar=True)
        return out["loss"]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
