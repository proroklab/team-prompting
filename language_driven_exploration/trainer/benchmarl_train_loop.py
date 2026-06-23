
"""
Launch several BenchMARL experiments sequentially.

"""

from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize

from trainer.benchmarl_setup_experiment_ import benchmarl_setup_experiment


# ---------------------------------------------------------------------------
# 1. List every training variant you want to run.
#    Each item is a perfectly normal Hydra override string.
#    Feel free to add / remove entries or build this list programmatically.
# ---------------------------------------------------------------------------
TRAINING_VARIANTS = [
    "+training=decentralized_language_driven_2",  # maps to conf/training/decentralized_language_driven_2.yaml
    "+training=decentralized_language_driven_3",
    "+training=decentralized_language_driven_4",
    "+training=decentralized_language_driven_5",
    "+training=decentralized_language_driven_6",
    "+training=decentralized_language_driven_7",
    "+training=decentralized_language_driven_8",
]

# Optional: choose where Hydra puts its output directories.
# Using ${hydra.job.name} keeps them distinct for each variant.
RUN_DIR_TEMPLATE = "outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"


def run_all():
    """Compose, build and run every experiment in TRAINING_VARIANTS sequentially."""
    conf_root = "../conf"   # ../conf relative to this script

    for override in TRAINING_VARIANTS:
        print(f"\n=== Launching experiment with override: {override} ===")

        # Every loop starts a fresh Hydra context so global state is reset.
        with initialize(config_path=conf_root, version_base=None):
            # Build the full config, applying our override *and* a unique run dir.
            cfg = compose(
                config_name="config",
                overrides=[
                    override,
                    f"hydra.run.dir={RUN_DIR_TEMPLATE}",          # put logs in a unique folder
                    f"hydra.job.name={override.split('=')[1]}",   # nicer folder names
                ],
            )

            print("-" * 60)
            print(OmegaConf.to_yaml(cfg, resolve=True))
            print("-" * 60)

            # Set up & run the experiment
            experiment = benchmarl_setup_experiment(cfg)
            experiment.run()           # blocking – when this returns the next job starts

        print(f"=== Finished: {override} ===\n")


if __name__ == "__main__":
    run_all()
