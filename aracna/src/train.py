import lightning as L
from omegaconf import DictConfig

from aracna.src.callbacks.registry import callback_registry
from aracna.src.callbacks.seqlen_warmup_reload import setup_seqlen_warmup
from aracna.src.datamodules import registry as datamodules_registry
from aracna.src.learning.aracna_train_object import AracnaTrain
from aracna.src.utils.config import get_object_from_registry, setup_wandb
from aracna.src.utils.constants import SEQLEN_REMOVE_KEYS


def get_data(task_info, task_config: DictConfig, loader_config):
    datamodule_getter, task_config_copy = get_object_from_registry(
        task_config, datamodules_registry, init=False, attr_ls=["dataset"]
    )
    return datamodule_getter(task_info, task_config_copy, loader_config)


def process_seqlen_trainer_config(config: DictConfig):
    trainer_dict = (
        config.trainer
        if config.trainer.seqlen_warmup is None
        else setup_seqlen_warmup(config)
    )
    return {k: v for k, v in trainer_dict.items() if k not in SEQLEN_REMOVE_KEYS}


def add_callbacks(config: DictConfig, trainer_dict: DictConfig):
    callback_ls = trainer_dict.get("callbacks", [])

    for callback_key, callback_config in config.callbacks.items():
        callback_ls.append(callback_registry[callback_key](**callback_config))

    return trainer_dict | {"callbacks": callback_ls}


def get_trainer(config: DictConfig):
    logger = setup_wandb(config) if config.get("wandb") is not None else None
    trainer_dict = process_seqlen_trainer_config(config)
    trainer_dict = add_callbacks(config, trainer_dict)
    return L.Trainer(**trainer_dict, logger=logger)


def train_from_config(config):
    model = AracnaTrain(config)
    # hparams and config usually the same, depends on task.
    trainer = get_trainer(model.hparams)
    datamodule = get_data(model.task_info, model.hparams.task, config.loader)
    trainer.fit(model, datamodule)
