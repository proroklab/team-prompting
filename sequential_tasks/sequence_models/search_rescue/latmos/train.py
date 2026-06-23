"""LATMOS training entry.

Smoke (3 epochs):
    PYTHONPATH=. python -m sequence_models.search_rescue.latmos.train --smoke

Full (200 epochs — model is small, converges fast):
    PYTHONPATH=. python -m sequence_models.search_rescue.latmos.train
"""

from __future__ import annotations

import argparse
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from sequence_models.search_rescue.latmos.data import make_loaders
from sequence_models.search_rescue.latmos.model import LatmosModel

DATASET_PATH = "sequence_models/data/search_rescue/dataset_search_rescue_ALL.json"
SAVE_DIR = pathlib.Path("sequence_models/search_rescue")
SAVE_PATH = SAVE_DIR / "latmos.pth"


def run_training(
    *,
    smoke: bool = False,
    batch_size: int = 64,
    neg_per_pos: int = 1,
    lr: float = 1e-4,
    use_wandb: bool = False,
) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        DATASET_PATH, batch_size=batch_size, neg_per_pos=neg_per_pos,
    )
    model = LatmosModel(lr=lr)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"LatmosModel constructed: {n_params:.2f}M params")

    run_name = f"latmos{'-smoke' if smoke else ''}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(SAVE_DIR / "latmos_checkpoints"),
        monitor="val_loss", mode="min", save_top_k=1,
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    logger = False
    if use_wandb and not smoke:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project="latmos", name=run_name, log_model=False)

    trainer = pl.Trainer(
        max_epochs=3 if smoke else 200,
        accelerator="cpu",
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    best_model = LatmosModel.load_from_checkpoint(best_ckpt)
    save_path = SAVE_PATH if not smoke else SAVE_DIR / "latmos_smoke.pth"
    torch.save(best_model.state_dict(), str(save_path))
    print(f"[{run_name}] best model saved → {save_path} (ckpt: {best_ckpt})")
    return str(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    run_training(smoke=args.smoke, use_wandb=args.wandb)
