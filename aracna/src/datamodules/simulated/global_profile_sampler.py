from aracna.src.datamodules.simulated.cna_real_profile_sampler import (
    RealProfileMixin,
    RealProfileSampler,
)
from aracna.src.datamodules.simulated.cna_sampler import PurelySimulated
from aracna.src.datamodules.simulated.sample_datamodule import BaseSampler
from aracna.src.task_info.cat_paired import SupervisedTrainInfo


class GlobalSampler(BaseSampler, RealProfileMixin):
    def __init__(
        self,
        profile_sampler: RealProfileSampler,
        task_info: SupervisedTrainInfo,
    ) -> None:
        # Prepend global predict vars at start
        # assume zeros for irrelevant global prediction i/o
        self.profile_sampler = profile_sampler
        self.task_info: SupervisedTrainInfo = task_info

    def set_curr_seqlen(self, curr_seqlen):
        self.profile_sampler.set_curr_seqlen(curr_seqlen)

    def sample(self, train_type):
        return self.profile_sampler.sample(train_type)

    @classmethod
    def real_from_config(cls, train_df, val_df, task_info, dataset_kwargs):
        profile_sampler = RealProfileSampler(
            train_df, val_df, task_info.base_info, **dataset_kwargs
        )
        return cls(profile_sampler, task_info)

    @classmethod
    def simmed_from_config(cls, task_info, dataset_kwargs):
        profile_sampler = PurelySimulated(task_info.base_info, **dataset_kwargs)
        return cls(profile_sampler, task_info)
