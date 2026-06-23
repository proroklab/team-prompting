import hydra
import os
import json
import re
import time
from pathlib import Path
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from trainer.benchmarl_setup_experiment_ import benchmarl_setup_experiment

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from torchrl.envs.utils import ExplorationType, set_exploration_type
from benchmarl.experiment import Experiment

@torch.no_grad()
def _evaluation_loop(self):
    evaluation_start = time.time()
    with set_exploration_type(
        ExplorationType.DETERMINISTIC
        if self.config.evaluation_deterministic_actions
        else ExplorationType.RANDOM
    ):
        if self.task.has_render(self.test_env) and self.config.render:
            video_frames = []

            def callback(env, td):
                video_frames.append(
                    self.task.__class__.render_callback(self, env, td)
                )

        else:
            video_frames = None
            callback = None

        if self.test_env.batch_size == ():
            rollouts = []
            for eval_episode in range(self.config.evaluation_episodes):
                rollouts.append(
                    self.test_env.rollout(
                        max_steps=self.max_steps,
                        policy=self.policy,
                        callback=callback if eval_episode == 0 else None,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                )
        else:
            rollouts = self.test_env.rollout(
                max_steps=self.max_steps,
                policy=self.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
                break_when_all_done=True,
                # We are running vectorized evaluation we do not want it to stop when just one env is done
            )
            rollouts = list(rollouts.unbind(0))
    evaluation_time = time.time() - evaluation_start
    self.logger.log(
        {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
    )
    self.logger.log_evaluation(
        rollouts,
        video_frames=video_frames,
        step=self.n_iters_performed,
        total_frames=self.total_frames,
    )
    # Callback
    self._on_evaluation_end(rollouts)


def _safe_filename(text: str, max_len: int = 32) -> str:
    """Return a filesystem-safe slug based on the first *max_len* characters."""
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", text[:max_len])
    return slug or "instruction"


NUM_ROLLOUTS = 100

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Interactive evaluation loop that encodes an instruction, runs the BenchMARL
    experiment, and saves a *single* image that contains **both** the occupancy
    heat-map and the team-spread time-series side-by-side.
    """

    # ---------------------------------------------------------------------
    # 1) Static settings that never change inside the loop
    # ---------------------------------------------------------------------
    restore_path = (
        "/path/to/language_driven_exploration/checkpoints/"
        "decentralized_no_language.pt"
    )

    cfg.experiment.restore_file = restore_path
    cfg.experiment.restore_map_location = "cpu"
    cfg.experiment.save_folder = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments"
    cfg.experiment.loggers[0] = "csv"
    print(Path(os.path.dirname(os.path.realpath(__file__))) / "experiments")
    cfg.experiment.render = True
    cfg.experiment.evaluation_episodes = NUM_ROLLOUTS
    cfg.task.params.done_at_termination = True
    
    #Experiment._evaluation_loop = _evaluation_loop

    print("Loaded Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # Pre-load the sentence-encoder once
    llm = SentenceTransformer("thenlper/gte-large", device="cpu")

    # Prepare deterministic directories relative to project root
    root_dir = get_original_cwd()
    data_dir = os.path.join(root_dir, "data")
    plots_dir = os.path.join(root_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    def bool_tag(name: str, value: bool) -> str:
            return name if value else f"no_{name}"
    
    config_tag = "__".join([
            bool_tag("gnn", cfg.task.params.use_gnn),
            bool_tag("conv2d", cfg.task.params.use_conv_2d),
            bool_tag("confidence", cfg.task.params.use_confidence_data),
        ])

    
    ckpt_file = os.path.basename(cfg.experiment.restore_file)
    ckpt_name = os.path.splitext(ckpt_file)[0]
    
    # Build nested directory: plots/{ckpt_name}/{flags}/
    config_dir = os.path.join(plots_dir, ckpt_name, config_tag)
    os.makedirs(config_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Interactive evaluation loop
    # ------------------------------------------------------------------
    eval_id = 0  # incremental counter for file names
    print("\nEnter instructions to evaluate (blank / 'quit' to stop).\n")

    new_sentence = input("Instruction > ").strip()
    if new_sentence.lower() in {"quit", "q", "exit"}:
        print("Exiting evaluation loop.")
        return

    # --------------------------------------------------------------
    # Encode sentence -> embedding (1D tensor)
    # --------------------------------------------------------------
    try:
        if new_sentence == "":
            embedding = torch.zeros(llm.get_sentence_embedding_dimension(), device="cpu")
            print("Using zero embedding for empty instruction.")
        else:
            embedding = torch.tensor(llm.encode([new_sentence]), device="cpu").squeeze(0)
    except Exception as e:
        print(f"Failed to encode instruction: {e}")
        return

    # --------------------------------------------------------------
    # Serialize to JSON (overwrite each run)
    # --------------------------------------------------------------
    json_path = os.path.join(data_dir, "evaluation_instruction.json")
    payload = {
        #"grid": [0.0] * 100,
        "gemini_response": new_sentence,
        "embedding": embedding.tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf)
    print(f"Saved instruction & embedding → {json_path}")

    # Tell the task where the freshly-written JSON lives
    cfg.task.params.data_json_path = json_path
    cfg.task.params.max_steps = 1000  # Set max steps for the task

    # --------------------------------------------------------------
    # Build experiment and run evaluation
    # --------------------------------------------------------------
    map_sizes = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    for map_size in map_sizes:
        cfg.task.params.x_semidim = map_size
        cfg.task.params.y_semidim = map_size

        print(f"Evaluating with map size: {map_size}m")
        cfg.experiment.save_folder = os.path.join(config_dir, f"map_{map_size:.1f}m")
        os.makedirs(cfg.experiment.save_folder, exist_ok=True)

            # Setup and run the experiment
        experiment = benchmarl_setup_experiment(cfg)
        experiment.evaluate()


if __name__ == "__main__":
    main()
