import warnings
from collections import defaultdict

import lightning as L
import torch

from aracna.src.learning.learning_tasks import registry as learning_task_registry
from aracna.src.learning.learning_tasks import scheduler_registry
from aracna.src.task_info import registry as task_info_registry
from aracna.src.utils.config import get_logger, get_object_from_registry

log = get_logger(__name__)


class AracnaTrain(L.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch.set_float32_matmul_precision("high")
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()

        # needs to be in this order as learning task might change config.
        self.model = learning_task_registry[config.task.name](config)

        self.task_info = get_object_from_registry(
            config.task.info, task_info_registry, init=True
        )

        self.save_hyperparameters(config, logger=False)
        # Passing in config expands it one level, so can access by
        # self.hparams.train instead of self.hparams.config.train

    def _log_dict(self, metrics, **kwargs):
        default_args = {
            "on_step": True,
            "on_epoch": True,
            "prog_bar": True,
            "add_dataloader_idx": False,
            "sync_dist": True,
        }
        default_args.update(kwargs)
        self.log_dict(metrics, **default_args)

    def step(self, batch, prefix="train", on_step=False):
        # TODO - kinda gross but make compatible with diff datamodules
        position_features, input, *_ = batch
        output = self.model(input, position_features)
        loss, metrics = self.trainer.datamodule.loss(batch, output, prefix=prefix)
        self._log_dict(metrics, on_step=on_step)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, on_step=True)
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self._log_dict(loss_epoch, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.step(batch, prefix="val")

    def get_scheduler(self, optimizer):
        lr_scheduler_cls, scheduler_hparams = get_object_from_registry(
            self.hparams.scheduler, scheduler_registry, init=False
        )
        interval = scheduler_hparams.pop("interval", "epoch")
        lr_scheduler = lr_scheduler_cls(optimizer, **scheduler_hparams)
        return {
            "scheduler": lr_scheduler,
            "interval": interval,  # 'epoch' or 'step'
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }

    def configure_optimizers(self):
        # get param groups
        params_dict = defaultdict(list)
        for p in self.parameters():

            def get_key(x):
                return "normal" if not x else frozenset(x.items())

            params_dict[get_key(getattr(p, "_optim", None))].append(p)

        # add param groups to optimizer
        optimizer = torch.optim.Adam(
            params_dict.pop("normal"), **self.hparams.optimizer
        )

        hp_list = [dict(hp) for hp in params_dict]
        print("Hyperparameter groups", hp_list)
        for hp, hp_group in params_dict.items():
            optimizer.add_param_group(
                {"params": hp_group, **self.hparams.optimizer, **dict(hp)}
            )

        # Print optimizer info for debugging
        unique_hparams = {k for hp in hp_list for k in hp}
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in unique_hparams}
            log.info(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )

        if self.hparams.scheduler is None:
            return optimizer
        lr_scheduler = self.get_scheduler(optimizer)

        return [optimizer], [lr_scheduler]

    def predict_step(self, batch):
        pos, input, sample_len, processed = batch
        max_trained_len = self.hparams_initial.trainer.seqlen_warmup[-1].seq_len
        if (sample_len > max_trained_len).any():
            warnings.warn("data size greater than model training", UserWarning)
        if not processed:
            pos, input, _ = self.task_info.process_inputs(
                pos, input, {"sample_len": sample_len}
            )

        # result = self.model.inference(input, pos)
        result = self.model(input, pos)
        return self.task_info.inference(pos, input, result, sample_len)
