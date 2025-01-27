import hydra
import matplotlib.pyplot as plt
from notebook_analyses.plotting_utils import get_read_depth_plot
from omegaconf import DictConfig
from aracna.src.datamodules.simulated.cna_real_profile_sampler import (
    RealProfileSampler,
)
from aracna.src.datamodules.simulated.global_profile_sampler import GlobalSampler
from aracna.src.learning.aracna_train_object import AracnaTrain
from aracna.src.utils.config import process_config
from aracna.src.train import get_data


def plot_outs(res, titles=None):
    if titles is None:
        titles = ["Reads", "Minor Allele Frequency", "Total C", "Minor C"]
    fig, axs = plt.subplots(len(titles) // 2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        ax.scatter(res[0], res[i + 1])
        ax.set_title(titles[i])


def test_real_sampler():
    config = {
        "max_seq_length": 1000,
        "data_path": "data/ReleasedData/TCGA_SNP6_hg38",
        "sample_n_chrom": 23,
        "prepend_len": 5,
    }
    sampler = RealProfileSampler.from_path(**config)
    out = sampler.sample("train")
    return


def test_global_sampler():
    config = {
        "max_seq_length": 1000,
        "data_path": "data/ReleasedData/TCGA_SNP6_hg38",
        "sample_n_chrom": 23,
        # "curr_seq_length": 1000,
        "prepend_len": 5,
    }
    sampler = GlobalSampler.from_path(**config)
    out = sampler.sample("train")
    return


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def test_datamodule(config: DictConfig):
    config = process_config(config)
    model = AracnaTrain(config)
    # hparams and config usually the same, depends on task.
    datamodule = get_data(model.task_info, model.hparams.task, config.loader)
    datamodule.set_curr_batch_size(1)
    get_read_depth_plot(
        15.0, purity=0.75, plot_n=10000, paired=True, tm=model, dm=datamodule
    )
    plt.show()
    return


if __name__ == "__main__":
    # test_real_sampler()
    # test_global_sampler()
    test_datamodule()
