import hydra
from omegaconf import DictConfig, OmegaConf

from aracna.configs import schemas


@hydra.main(config_path="../configs", config_name="config")
def test_config_files(config: OmegaConf):
    # for this to work need to remove default base_schema from config.yaml
    assert config.trainer
    assert config.dataset
    assert config.model
    assert config.wandb
    assert config.optimizer
    assert config.scheduler


def test_config_w_schema():
    @hydra.main(
        version_base=None,
        config_path="../configs",
        config_name="config",
    )
    def inner(_cfg: DictConfig) -> None:
        cfg: schemas.Config = OmegaConf.to_object(
            _cfg
        )  # __post_init__ runs here
        print(OmegaConf.to_yaml(cfg))

    inner()


if __name__ == "__main__":
    # OmegaConf.register_new_resolver('eval', eval)
    # test_config_files()
    test_config_w_schema()
