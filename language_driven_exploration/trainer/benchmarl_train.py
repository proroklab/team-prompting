import hydra
from trainer.benchmarl_setup_experiment_ import benchmarl_setup_experiment
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))   # full merged config

    experiment = benchmarl_setup_experiment(cfg)
    experiment.run()     

if __name__ == "__main__":
    main()