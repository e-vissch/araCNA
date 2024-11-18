import abc
from functools import cached_property, partial

import lightning as L

from aracna.src.metrics.metrics import output_metric_fns


class TaskInfoModule(L.LightningDataModule, abc.ABC):
    def __init__(self, config, task_info):
        super().__init__()

        self.loss_name = config.loss
        self.loss_kwargs = config.loss_kwargs
        self.config_metrics = config.metrics

        self.task_info = task_info  # type: ignore

    @cached_property
    def loss_func(self):
        loss_func = getattr(
            self.task_info, self.loss_name, output_metric_fns.get(self.loss_name)
        )
        if loss_func is None:
            error =  f"Loss function {self.loss_name} not found in task_info\
                  or output_metric_fns"
            raise ValueError(error)
        return partial(loss_func, **self.loss_kwargs)

    def loss(self, *args, prefix="train"):
        kwargs = self.task_info.process_for_loss(*args)
        loss = self.loss_func(**kwargs)
        metrics = self.metrics(prefix=prefix, **kwargs)
        return loss, metrics | {f"{prefix}/loss": loss}

    @cached_property
    def metric_names(self):
        return {
            metric: getattr(
                self.task_info,
                metric,
                getattr(self, metric, output_metric_fns.get(metric)),
            )
            for metric in self.config_metrics
        }

    def metrics(self, *args, prefix=None, **kwargs):
        return self.task_info.metrics(self.metric_names, *args, prefix=prefix, **kwargs)
