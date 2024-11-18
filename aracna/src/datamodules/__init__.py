from functools import partial

from .simulated.cna_real_profile_sampler import RealProfileSampler
from .simulated.global_profile_sampler import GlobalSampler
from .simulated.sample_datamodule import ProbDataModule


def get_sampler(class_method):
    return partial(ProbDataModule, class_method)


probability_registry = {
    "real_just_seq": get_sampler(RealProfileSampler.real_from_config),
    "real_simmed_global": get_sampler(GlobalSampler.from_path),
    "simmed_global": get_sampler(GlobalSampler.simmed_from_config),
}

registry = probability_registry
