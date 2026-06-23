"""Training entry for the distilled-SLM baseline.

Mirrors ``sequence_models/search_rescue/model_training/rnn_model.py`` so the
downstream comparison is on identical data + supervision.

Smoke (3 epochs):
    PYTHONPATH=. python -m sequence_models.search_rescue.distilled_slm.train --smoke

Full (100 epochs — pretrained model converges fast):
    PYTHONPATH=. python -m sequence_models.search_rescue.distilled_slm.train
"""

from __future__ import annotations

import argparse
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from sequence_models.search_rescue.distilled_slm.data import make_loaders
from sequence_models.search_rescue.distilled_slm.slm_model import (
    DistilledSLM, load_tokenizer, DEFAULT_MODEL_NAME, N_LAYERS_KEEP,
)

DATASET_PATH = "sequence_models/data/search_rescue/dataset_search_rescue_ALL.json"
SAVE_DIR = pathlib.Path("sequence_models/search_rescue")
SAVE_PATH = SAVE_DIR / "distilled_slm.pth"


def run_training(
    *,
    smoke: bool = False,
    batch_size: int = 32,
    positions_per_trace: int = 5,
    max_seq_len: int = 256,
    lr: float = 5e-5,
    use_wandb: bool = False,
) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    train_loader, val_loader = make_loaders(
        DATASET_PATH, tokenizer,
        batch_size=batch_size,
        positions_per_trace=positions_per_trace,
        max_seq_len=max_seq_len,
    )

    model = DistilledSLM(
        model_name=DEFAULT_MODEL_NAME,
        n_layers_keep=N_LAYERS_KEEP,
        lr=lr,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DistilledSLM constructed: {n_params:.2f}M params, {N_LAYERS_KEEP} encoder layers kept")

    run_name = f"distilled_slm{'-smoke' if smoke else ''}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(SAVE_DIR / "slm_checkpoints"),
        monitor="val_loss", mode="min", save_top_k=1,
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    logger = False
    if use_wandb and not smoke:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project="distilled-slm", name=run_name, log_model=False)

    trainer = pl.Trainer(
        max_epochs=3 if smoke else 100,
        accelerator="cpu",  # match the RNN training device for reproducibility
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    best_model = DistilledSLM.load_from_checkpoint(best_ckpt)
    save_path = SAVE_PATH if not smoke else SAVE_DIR / "distilled_slm_smoke.pth"
    torch.save(best_model.state_dict(), str(save_path))
    print(f"[{run_name}] best model saved → {save_path} (ckpt: {best_ckpt})")
    return str(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    run_training(smoke=args.smoke, use_wandb=args.wandb)
