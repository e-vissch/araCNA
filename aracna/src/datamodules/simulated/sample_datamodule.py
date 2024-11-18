from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import torch
from aracna.src.datamodules.metric_datamodule import TaskInfoModule
from aracna.src.task_info.cat_paired import SupervisedTrainInfo
from aracna.src.task_info.task_info import SeqInfo
from torch.utils.data import DataLoader, IterableDataset


@dataclass
class BaseSampler(abc.ABC):
    task_info: SeqInfo | SupervisedTrainInfo

    def sample(self):
        return NotImplementedError

    def sample_torch_simple(
        self,
        train_type="train",
        float_dtypes=torch.float32,
        int_dtypes=torch.int32,
    ):
        pos_feats, inputs, targets, metric_kwargs, global_targets = self.sample(
            train_type
        )

        def get_correct_dtype(val):
            if isinstance(val, float):
                return torch.tensor(val, dtype=float_dtypes)
            return val

        if self.task_info.targets_included:
            return (
                torch.tensor(np.stack(pos_feats, axis=-1), dtype=int_dtypes),
                torch.tensor(np.stack(inputs, axis=-1), dtype=float_dtypes),
                torch.tensor(np.stack(targets, axis=-1), dtype=float_dtypes),
                {key: torch.tensor(val) for key, val in metric_kwargs.items()},
                {key: get_correct_dtype(glob) for key, glob in global_targets.items()},
            )

        return (
            torch.tensor(np.stack(pos_feats, axis=-1), dtype=int_dtypes),
            torch.tensor(np.stack(inputs, axis=-1), dtype=float_dtypes),
            {key: torch.tensor(val) for key, val in metric_kwargs.items()},
        )

    def sample_torch(
        self, train_type="train", float_dtypes=torch.float32, int_dtypes=torch.int32
    ):
        return self.task_info.process_inputs(
            *self.sample_torch_simple(train_type, float_dtypes, int_dtypes)
        )

    def set_curr_seqlen(self):
        raise NotImplementedError


class SharedProbDataset(IterableDataset):
    def __init__(
        self,
        sampler: BaseSampler,
        float_dtypes=torch.float32,
        int_dtypes=torch.int32,
        sampler_kwargs=None,
        n_batches=100,
    ):
        # because dataset is infinite, we need to specify how many batches to sample
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.dtypes = {"float_dtypes": float_dtypes, "int_dtypes": int_dtypes}
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.sampler.sample_torch(**self.dtypes, **self.sampler_kwargs)


class ProbDataModule(TaskInfoModule):
    def __init__(self, sampler_factory, task_info, task_config, loader_config) -> None:
        super().__init__(task_config, task_info)

        dataset_config = task_config.dataset.copy()
        n_batches = dataset_config.pop("n_batches")

        self.sampler: BaseSampler = sampler_factory(task_info, dataset_config)

        self.train_dataset = SharedProbDataset(self.sampler, n_batches=n_batches)
        self.val_dataset = SharedProbDataset(
            self.sampler, sampler_kwargs={"train_type": "val"}, n_batches=n_batches
        )
        self.loader_config = loader_config

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.loader_config)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.loader_config)

    def set_curr_seqlen(self, curr_seqlen):
        self.sampler.set_curr_seqlen(curr_seqlen)

    def set_curr_batch_size(self, batch_size):
        self.loader_config["batch_size"] = batch_size
