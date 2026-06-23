"""Distilled-SLM model: truncated flan-t5-small encoder + h/state heads.

Loads ``google/flan-t5-small`` (60M total / 35.3M encoder), keeps the shared
embedding + first 4 of 8 encoder layers (~25.9M params) to roughly match
the parameter budget of the EventRNN baseline. Adds two linear heads:

* ``h_head``     : pooled encoder hidden → 1024-dim subtask embedding (gte-large-shaped)
* ``state_head`` : pooled encoder hidden → STATE_DIM logits

Training mirrors the EventRNN pipeline (cosine on h, BCE on state) for an
apples-to-apples comparison.
"""

from __future__ import annotations

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import T5EncoderModel, T5Tokenizer

H_DIM = 1024  # matches gte-large embedding dim used by the augmented dataset
STATE_DIM = 10  # matches search_rescue: 2 aut bits + 4 + 4 sub-state bits
DEFAULT_MODEL_NAME = "google/flan-t5-small"
N_LAYERS_KEEP = 4  # 4 of 8 → ~25.9M params (≈ "param-matched" target of 25M)


def load_truncated_encoder(model_name: str = DEFAULT_MODEL_NAME, n_layers_keep: int = N_LAYERS_KEEP) -> T5EncoderModel:
    enc = T5EncoderModel.from_pretrained(model_name)
    full = len(enc.encoder.block)
    if not (1 <= n_layers_keep <= full):
        raise ValueError(f"n_layers_keep must be in [1,{full}], got {n_layers_keep}")
    enc.encoder.block = enc.encoder.block[:n_layers_keep]
    enc.config.num_layers = n_layers_keep
    return enc


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(model_name)


def _masked_mean_pool(last_hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool over valid tokens. last_hidden: (B,T,D), mask: (B,T)."""
    m = attention_mask.unsqueeze(-1).float()  # (B,T,1)
    s = (last_hidden * m).sum(dim=1)
    d = m.sum(dim=1).clamp(min=1e-6)
    return s / d


class DistilledSLM(pl.LightningModule):
    """Truncated flan-t5-small encoder with regression + classification heads.

    Public interface mirrors the EventRNN's downstream contract:
      * ``encode(input_ids, attention_mask) -> (h_pred, state_logits)``
      * ``state_logits`` is shape (B, STATE_DIM); apply sigmoid for prob.
      * ``h_pred`` is shape (B, H_DIM); used as the subtask embedding.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        n_layers_keep: int = N_LAYERS_KEEP,
        h_dim: int = H_DIM,
        state_dim: int = STATE_DIM,
        lr: float = 5e-5,
        cls_loss_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = load_truncated_encoder(model_name, n_layers_keep)
        d_model = self.encoder.config.d_model

        self.h_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, h_dim),
        )
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, state_dim),
        )
        self.lr = lr
        self.cls_w = cls_loss_weight

    # ------------------------------------------------------------------ #

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _masked_mean_pool(out.last_hidden_state, attention_mask)
        h_pred = self.h_head(pooled)
        state_logits = self.state_head(pooled)
        return h_pred, state_logits

    # ------------------------------------------------------------------ #

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        h_pred, state_logits = self.encode(batch["input_ids"], batch["attention_mask"])
        target_h = batch["target_h"]
        target_label = batch["target_label"]

        cos = F.cosine_similarity(h_pred, target_h, dim=-1)  # (B,)
        loss_recon = (1 - cos).mean()

        loss_cls = F.binary_cross_entropy_with_logits(state_logits, target_label).mean()

        loss = self.cls_w * loss_cls + loss_recon
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "cos_mean": cos.mean()}

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log_dict({f"train_{k}": v for k, v in out.items()}, prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in out.items()}, prog_bar=True)
        return out["loss"]

    def configure_optimizers(self):
        # Two-group LR: lower for the pre-trained encoder, higher for fresh heads.
        encoder_params = list(self.encoder.parameters())
        head_params = list(self.h_head.parameters()) + list(self.state_head.parameters())
        return torch.optim.AdamW([
            {"params": encoder_params, "lr": self.lr},
            {"params": head_params,    "lr": self.lr * 5},
        ], weight_decay=0.01)
