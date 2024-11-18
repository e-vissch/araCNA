import hydra
from omegaconf import DictConfig
from aracna.src.train import train_from_config
from aracna.src.utils.config import process_config


@hydra.main(
    version_base=None, config_path="../configs", config_name="train_config.yaml"
)
def main(config: DictConfig):
    config = process_config(config)
    train_from_config(config)


if __name__ == "__main__":
    main()
