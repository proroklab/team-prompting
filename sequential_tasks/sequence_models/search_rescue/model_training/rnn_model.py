"""RNN training entry point for the Search-and-Rescue task.

Reuses the ``EventRNN`` LightningModule from ``sequence_models.four_flags.model_training.rnn_model``
but with task-specific constants (event_dim=7, state_dim=10, max_seq_len=20)
and dataset path. We override the dataset class locally so we can use our
own MAX_SEQ_LEN without monkey-patching the four_flags module.

Smoke test (a few epochs, no wandb):
    PYTHONPATH=. python -m sequence_models.search_rescue.model_training.rnn_model --smoke

Full run:
    PYTHONPATH=. python -m sequence_models.search_rescue.model_training.rnn_model
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

# Reuse the EventRNN module — its forward signatures are model-architecture-only,
# they don't depend on the four_flags constants.
from sequence_models.four_flags.model_training.rnn_model import EventRNN

# ─────────────────────────── Task constants ─────────────────────────── #

MAX_SEQ_LEN = 20
EVENT_DIM = 7
STATE_DIM = 10
SUBTASK_DECODER_DIM = 8
NUM_AUTOMATA = 4

DATASET_PATH = "sequence_models/data/search_rescue/dataset_search_rescue_ALL.json"
SAVE_DIR = pathlib.Path("sequence_models/search_rescue")
SAVE_PATH = SAVE_DIR / "search_rescue.pth"

# ─────────────────────────── Dataset ─────────────────────────── #


class SearchRescueEventDataset(Dataset):
    """Mirrors four_flags' EventSequenceDataset but uses our MAX_SEQ_LEN."""

    def __init__(self, raw: List[Dict]):
        self.samples: List[Dict[str, Tensor]] = []
        for ix, s in enumerate(raw):
            e_key = "events" if "events" in s else "e"
            required = (e_key, "y", "h", "label", "subtask_decoder_labels")
            for k in required:
                if k not in s:
                    raise KeyError(f"Sample {ix} missing key {k!r}")
            self.samples.append({
                "e": torch.tensor(s[e_key], dtype=torch.float32),
                "y": torch.tensor(s["y"], dtype=torch.float32),
                "h": torch.tensor(s["h"], dtype=torch.float32),
                "label": torch.tensor(s["label"], dtype=torch.float32),
                "subtask_label": torch.tensor(s["subtask_decoder_labels"], dtype=torch.float32),
            })

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "SearchRescueEventDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("JSON must be an array of samples")
        return cls(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    @staticmethod
    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Pad to MAX_SEQ_LEN. Matches four_flags' collate exactly:
        e is length T → padded to max_len; h/label/subtask include the
        initial-state row so length is T+1 → padded to max_len+1. EventRNN's
        ``_step`` slices off the initial row at training time.
        """
        lengths = torch.tensor([s["e"].size(0) for s in batch])
        max_len = MAX_SEQ_LEN
        event_dim = batch[0]["e"].size(1)
        h_dim = batch[0]["h"].size(1)
        label_dim = batch[0]["label"].size(1)
        subtask_dim = batch[0]["subtask_label"].size(1)

        e_out, h_out, y_out, labels, subtask_labels = [], [], [], [], []
        for s in batch:
            T = s["e"].size(0)
            pad_e = torch.zeros((max_len - T, event_dim))
            pad_h = torch.zeros((max_len - T, h_dim))
            pad_label = torch.zeros((max_len - T, label_dim))
            pad_subtask = torch.zeros((max_len - T, subtask_dim))

            e_out.append(torch.cat([s["e"], pad_e]))
            h_out.append(torch.cat([s["h"], pad_h]))
            labels.append(torch.cat([s["label"], pad_label]))
            subtask_labels.append(torch.cat([s["subtask_label"], pad_subtask]))
            y_rep = s["y"].unsqueeze(0).expand(max_len, -1)
            y_out.append(y_rep)

        return {
            "e": torch.stack(e_out),
            "h": torch.stack(h_out),
            "y": torch.stack(y_out),
            "label": torch.stack(labels),
            "subtask_labels": torch.stack(subtask_labels),
            "lengths": lengths,
        }


def make_loaders(json_path: str, batch_size: int = 128, train_frac: float = 0.8, *, seed: int = 42):
    full_ds = SearchRescueEventDataset.from_json(json_path)
    n_total = len(full_ds)
    n_train = math.floor(n_total * train_frac)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, (n_train, n_val), generator=gen)
    kwargs = dict(
        batch_size=batch_size,
        collate_fn=SearchRescueEventDataset.collate,
        num_workers=0,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
    )


# ─────────────────────────── Training ─────────────────────────── #


def run_training(
    *,
    smoke: bool = False,
    batch_size: int = 128,
    input_dim: int = 64,
    resume_ckpt: str | None = None,
    use_wandb: bool = False,
) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(DATASET_PATH, batch_size=batch_size)

    model = EventRNN(
        lr=5e-5,
        event_dim=EVENT_DIM,
        y_dim=1024,
        latent_dim=1024,
        state_dim=STATE_DIM,
        input_dim=input_dim,
        num_layers=1,
        cls_loss_weight=8.0,
        ground_truth_h_rate=0.0,
        decoder=None,
        use_encoder=False,
    )

    run_name = f"search_rescue-bs{batch_size}-in{input_dim}{'-smoke' if smoke else ''}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(SAVE_DIR / "checkpoints"),
        monitor="val_loss", mode="min", save_top_k=1,
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    logger = False
    if use_wandb and not smoke:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project="event-rnn", name=run_name, log_model=False)

    trainer = pl.Trainer(
        max_epochs=20 if smoke else 2000,
        accelerator="cpu",  # GPU has CUDA-kernel mismatch with torch 2.6 prebuilts
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)

    best_ckpt = checkpoint_cb.best_model_path
    best_model = EventRNN.load_from_checkpoint(best_ckpt)
    save_path = SAVE_PATH if not smoke else SAVE_DIR / "search_rescue_smoke.pth"
    torch.save(best_model.state_dict(), str(save_path))
    print(f"[{run_name}] best model saved → {save_path} (ckpt: {best_ckpt})")
    return str(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Short smoke-test run (20 epochs)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    run_training(smoke=args.smoke, resume_ckpt=args.resume, use_wandb=args.wandb)
