import hydra
from omegaconf import DictConfig

from aracna.src.utils.config import process_config
from aracna.analysis.sim_inference_utils import get_simulated_infer
from aracna.src.train import train_from_config


@hydra.main(
    version_base=None, config_path="../configs", config_name="train_config.yaml"
)
def small_run(config: DictConfig):
    config = process_config(config)

    ## overrides:
    config["wandb"] = None
    config["loader"]["batch_size"] = 2
    config["task"]["info"]["max_seq_length"] = 10000
    train_from_config(config)


@hydra.main(
    version_base=None, config_path="../configs", config_name="train_config.yaml"
)
def small_run_gpu(config: DictConfig):
    config = process_config(config)

    ## overrides:
    config["trainer"]["accelerator"] = "gpu"
    config["wandb"] = None
    config["task"]["info"]["max_seq_length"] = 10000

    train_from_config(config)



def test_sim_infer(model_key, read_depth=15, purity=0.87):
    get_simulated_infer(model_key, read_depth, purity)



if __name__ == "__main__":
    small_run()
    # small_run_gpu() # should work on cpu
    # test_sim_infer("pjflljt4") # hyena model should work on cpu
