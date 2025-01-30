from dataclasses import dataclass
import os
from typing import Optional

import lightning as L
import torch
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader

from aracna.src.learning.aracna_train_object import AracnaTrain
from aracna.src.learning.learning_tasks import registry as learning_task_registry
from aracna.src.task_info.cat_paired import PairedInfo
from aracna.src.train import get_data
from aracna.src.utils.config import package_model_path


@dataclass
class InferenceInfo:
    code_name: str
    trained_model: AracnaTrain
    trainer: Trainer
    data_module: Optional[LightningDataModule] = None

    @property
    def task_info(self):
        return self.trained_model.task_info

    @property
    def max_trained_tot_CN(self):
        return self.trained_model.hparams.task.dataset.get("max_total", 6)

    @property
    def can_train_upto_tot_CN(self):
        return self.task_info.max_tot_cn

    def get_prediction(self, pos, inp, sample_len, processed=False):
        return self.trainer.predict(
            self.trained_model,
            dataloaders=[
                DataLoader(
                    list(zip(pos, inp, sample_len, [processed for _ in pos])),
                    batch_size=1,
                )
            ],
        )[0]


def get_trainer():
    acltr = "gpu" if torch.cuda.is_available() else "cpu"
    return Trainer(accelerator=acltr, enable_progress_bar=False)


def amend_config_for_infer(config):
    for k, _ in config.task.info.items():
        if k not in PairedInfo.__dataclass_fields__ and k != "name":
            del config.task.info[k]

    config.task.info.name = f"infer_{config.task.info.name}"


def get_checkpointed_model(model_checkpoint, infer=True, task=None):
    # AracnaTrain.load_from_checkpoint creates recurrance if model has been iteratively
    # retrained, hence why we use this approach.
    model_path = package_model_path(model_checkpoint)
    trained_model_info = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False) 
    # lightning will put module to device later, so always construct on cpu
    config = trained_model_info["hyper_parameters"]
    config.task.model_checkpoint = model_checkpoint  # get same config.
    if task is not None:
        config.task.name = task

    if config.task.dataset.get("max_total") is None:
        config.task.dataset.max_total = 6

    # remove legacy keys:
    for key in ["neg_entropy_weight", "small_CN_weight"]:
        if key in config.task.info:
            del config.task.info[key]
    for key in ["inject_high_cn", "inject_rd"]:
        if key in config.task.dataset:
            del config.task.dataset[key]
    if config.task.name not in learning_task_registry.keys():
        print("Task name not recognised. Assuming legacy name, changing to pretrained")
        config.task.name = "pretrained"

    if infer:
        amend_config_for_infer(config)
    return AracnaTrain(config)


def infer(config):
    model = get_checkpointed_model(config.model_checkpoint)
    trainer = L.Trainer(**config.trainer)
    datamodule = get_data(model.task_info, model.hparams.task, config.loader)
    trainer.predict(model, datamodule)
