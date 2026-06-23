"""
Train sentence-decoder models with three size variants (small/medium/large) for each embedding model
and log all runs to Weights & Biases.

• small  – 1 hidden layer, 128 units (original architecture)
• medium – 2 hidden layers, 128 units each
• large  – 2 hidden layer, 256 units each

The script loops automatically over every LLM × model‑size combination and stores the best
model checkpoints plus a summary of best validation losses.
"""

import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from sentence_transformers import SentenceTransformer

from process_gemini_data import (
    collect_train_test_data_grid_attribute_confidence,
    collect_train_test_data_from_embeddings_confidence,
)

# ---------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------

PROJECT_NAME = "sentence_decoder_final"
JSON_DATA_FILE = "sentences/gemini_patch_dataset_scale_target_color_confidence.json"
# JSON_DATA_FILE = "data/language_data_complete_multi_target_color_scale_confidence.json"

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
SEED = 42
BATCH_SIZE = 128
EPOCHS = 750
LEARNING_RATE = 1e-4

# task‑specific weights (alpha: grid, beta: confidence, gamma: class, theta: target)
ALPHA = BETA = GAMMA = THETA = 1.0

USE_CONFIDENCE = True
USE_CLASS = True
USE_TARGET = True

NUM_CONFIDENCE = 3  # confidence classes
NUM_CLASS = 5       # class classes
NUM_TARGETS = 5     # target classes

# ---------------------------------------------------------------------------
# Utility ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Make runs deterministic (as far as possible)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(x, num_classes=num_classes).float()


# ---------------------------------------------------------------------------
# Decoder ----------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """Flexible MLP decoder that supports an arbitrary list of hidden sizes."""

    def __init__(self, in_size: int, out_size: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


# Mapping from size label → hidden‑layer configuration
MODEL_SIZES: dict[str, list[int]] = {
    "small": [128],        # original architecture
    "medium": [128, 128],  # extra hidden layer
    "large": [256, 256],        # wider hidden layer
}


# ---------------------------------------------------------------------------
# Loss -------------------------------------------------------------------------

bce_logits = nn.BCEWithLogitsLoss()

def loss_fn(
    model: nn.Module,
    emb: torch.Tensor,
    grid: torch.Tensor,
    confidence: torch.Tensor,
    class_goal: torch.Tensor,
    target_goal: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    theta: float = 1.0,
):
    """Composite loss that optionally includes confidence, class and target components."""

    pred = model(emb)
    offset = grid.shape[1]

    pred_grid = pred[:, :offset]
    total_loss = alpha * bce_logits(pred_grid, grid)
    grid_loss = total_loss.item()

    losses = [grid_loss]

    if USE_CONFIDENCE:
        pred_conf = pred[:, offset : offset + NUM_CONFIDENCE]
        confidence_loss = beta * bce_logits(pred_conf, confidence)
        total_loss += confidence_loss
        offset += NUM_CONFIDENCE
        losses.append(confidence_loss.item())
    else:
        losses.append(0.0)

    if USE_CLASS:
        pred_class = pred[:, offset : offset + NUM_CLASS]
        class_loss = gamma * bce_logits(pred_class, class_goal)
        total_loss += class_loss
        offset += NUM_CLASS
        losses.append(class_loss.item())
    else:
        losses.append(0.0)

    if USE_TARGET:
        pred_target = pred[:, offset : offset + NUM_TARGETS]
        target_loss = theta * bce_logits(pred_target, target_goal)
        total_loss += target_loss
        offset += NUM_TARGETS
        losses.append(target_loss.item())
    else:
        losses.append(0.0)

    return total_loss, *losses  # (total, grid, conf, class, target)


# ---------------------------------------------------------------------------
# Embedding models (LLMs) -------------------------------------------------------

LLMS = {
    SentenceTransformer("WhereIsAI/UAE-Large-V1"): "WhereIsAI/UAE-Large-V1",
    SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=None): "avsolatorio/GIST-large-Embedding-v0",
    SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1"): "mixedbread-ai/mxbai-embed-large-v1",
    SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True): "jinaai/jina-embeddings-v3",
    SentenceTransformer("BAAI/bge-large-en-v1.5"): "BAAI/bge-large-en-v1.5",
    SentenceTransformer("hkunlp/instructor-large"): "hkunlp/instructor-large",
    SentenceTransformer("thenlper/gte-large"): "thenlper/gte-large",
}

# ---------------------------------------------------------------------------
# Training loop ----------------------------------------------------------------

def train_single_run(
    *,
    llm_name: str,
    hidden_sizes: list[int],
    train: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    output_dim: int,
):
    set_seed(SEED)

    run = wandb.init(
        project=PROJECT_NAME,
        name=f"{llm_name}_{'x'.join(map(str, hidden_sizes))}_{datetime.now().isoformat(timespec='seconds')}",
        config=dict(
            llm=llm_name,
            model_sizes=hidden_sizes,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        ),
    )

    model = MLPDecoder(train["task_embedding"].shape[1], output_dim, hidden_sizes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    pbar = tqdm.tqdm(total=EPOCHS * (len(train["task_embedding"]) // BATCH_SIZE), leave=False)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        for i in range(len(train["task_embedding"]) // BATCH_SIZE):
            idx = slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)

            emb = torch.tensor(train["task_embedding"][idx], dtype=torch.float32).to(DEVICE)
            goal = torch.tensor(train["goal"][idx], dtype=torch.float32).to(DEVICE)
            conf = torch.tensor(train["confidence"][idx], dtype=torch.long).to(DEVICE) + 1
            class_goal = torch.tensor(train["class"][idx], dtype=torch.long).to(DEVICE) + 1
            target_goal = torch.tensor(train["max_targets"][idx], dtype=torch.float32).to(DEVICE)

            conf_oh = one_hot(conf, NUM_CONFIDENCE)
            class_oh = one_hot(class_goal, NUM_CLASS)
            target_oh = one_hot(target_goal.long(), NUM_TARGETS)

            optimizer.zero_grad()
            loss, grid_loss, conf_loss, class_loss, target_loss = loss_fn(
                model,
                emb,
                goal,
                conf_oh,
                class_oh,
                target_oh,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
                theta=THETA,
            )
            loss.backward()
            optimizer.step()

            # ------------------ validation ------------------
            with torch.no_grad():
                v_emb = torch.tensor(test["task_embedding"], dtype=torch.float32).to(DEVICE)
                v_goal = torch.tensor(test["goal"], dtype=torch.float32).to(DEVICE)
                v_conf = torch.tensor(test["confidence"], dtype=torch.long).to(DEVICE) + 1
                v_class = torch.tensor(test["class"], dtype=torch.long).to(DEVICE) + 1
                v_target = torch.tensor(test["max_targets"], dtype=torch.float32).to(DEVICE)

                v_conf_oh = one_hot(v_conf, NUM_CONFIDENCE)
                v_class_oh = one_hot(v_class, NUM_CLASS)
                v_target_oh = one_hot(v_target.long(), NUM_TARGETS)

                val_loss, v_grid, v_conf, v_cls, v_tgt = loss_fn(
                    model,
                    v_emb,
                    v_goal,
                    v_conf_oh,
                    v_class_oh,
                    v_target_oh,
                    alpha=ALPHA,
                    beta=BETA,
                    gamma=GAMMA,
                    theta=THETA,
                )
                best_val_loss = min(best_val_loss, val_loss.item())

            # ------------------ logging ---------------------
            pbar.update()
            pbar.set_description(
                f"{llm_name} | {'/'.join(map(str, hidden_sizes))} | epoch {epoch} | "
                f"loss {loss.item():.4f} val {val_loss.item():.4f}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "grid_loss": grid_loss,
                    "confidence_loss": conf_loss,
                    "class_loss": class_loss,
                    "target_loss": target_loss,
                    "val_loss": val_loss.item(),
                    "val_grid_loss": v_grid,
                    "val_confidence_loss": v_conf,
                    "val_class_loss": v_cls,
                    "val_target_loss": v_tgt,
                    "best_val_loss": best_val_loss,
                }
            )

    pbar.close()
    # save checkpoint -----------------------------------------------------------
    ckpt_name = f"{llm_name.replace('/', '_')}_{'x'.join(map(str, hidden_sizes))}_decoder.pt"
    torch.save(model.state_dict(), ckpt_name)
    wandb.finish()
    return best_val_loss, ckpt_name


# ---------------------------------------------------------------------------
# Main -------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(SEED)

    all_results: dict[str, float] = {}

    for llm, llm_name in LLMS.items():
        print(f"Preparing data for {llm_name} …")
        train, test = collect_train_test_data_grid_attribute_confidence(
            llm=llm,
            json_path=JSON_DATA_FILE,
            train_ratio=0.8,
            test_ratio=0.2,
            device=DEVICE,
        )

        # Compute output dimensionality (grid + optional heads)
        output_dim = train["goal"].shape[1]
        if USE_CONFIDENCE:
            output_dim += NUM_CONFIDENCE
        if USE_CLASS:
            output_dim += NUM_CLASS
        if USE_TARGET:
            output_dim += NUM_TARGETS

        for size_label, hidden_layers in MODEL_SIZES.items():
            key = f"{llm_name}_{size_label}"
            print(f"\n=== Training {key} ===")
            best_loss, ckpt = train_single_run(
                llm_name=llm_name,
                hidden_sizes=hidden_layers,
                train=train,
                test=test,
                output_dim=output_dim,
            )
            all_results[key] = best_loss
            print(f"Finished {key}: best val_loss = {best_loss:.4f}")

    print("\n=== Summary of best validation losses ===")
    for k, v in all_results.items():
        print(f"{k:60} {v:.4f}")