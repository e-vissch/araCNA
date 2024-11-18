import logging
import os
import warnings
from typing import Any
import zipfile

from aracna.configs import schemas
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.errors import CommError


def get_wandb_logger(final_conf):
    return WandbLogger(
        config=OmegaConf.to_container(final_conf, resolve=True),
        settings=wandb.Settings(start_method="fork"),  # type: ignore
        **final_conf.wandb,
    )


def setup_wandb(config: DictConfig):
    # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
    # Can pass in config_exclude_keys='wandb' to remove certain groups
    try:
        return get_wandb_logger(config)

    except CommError:
        config.wandb.mode = "offline"
        return get_wandb_logger(config)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def allow_mutable_config_with_pop(config: DictConfig | Any):
    if isinstance(config, DictConfig):
        # Set struct mode to False for the current config
        OmegaConf.set_struct(config, False)
        # Recursively apply for all nested configs
        for _, value in config.items():
            allow_mutable_config_with_pop(value)


def process_config(config: DictConfig) -> DictConfig:  #
    # validate config, but want as dict, probably a better way to do this?
    _config: schemas.Config = OmegaConf.to_object(config)

    log = get_logger()

    # enable adding new keys to config
    allow_mutable_config_with_pop(config)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Just want dict now that it has been validated
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def get_object_from_registry(config, registry, init=True, attr_ls=()):
    config_copy = config.copy()

    to_pop = config_copy
    for attr in attr_ls:
        to_pop = getattr(to_pop, attr)

    if init:
        return registry[to_pop.pop("name")](**config_copy)

    return registry[to_pop.pop("name")], config_copy


def package_model_path(model_path):
    if os.path.exists(given_path :=  model_path):
        return given_path
    
    rel_path = f"{os.path.dirname(__file__)}/../../"
    if os.path.exists(global_rel_path :=  f"{rel_path}{model_path}"):
        print("Using path relative to package install for saved model.")
        return global_rel_path
    
    # kinda gross but only have to unzip once
    if os.path.exists(zip_path :=  f"{rel_path}araCNA-models.zip") and not os.path.exists(f"{rel_path}araCNA-models"):
        # Unzip the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(rel_path)
        # recheck for path now extracted
        if os.path.exists(global_rel_path):
            print("Using path relative to package install for saved model.")
            return global_rel_path  
    else:
        raise FileNotFoundError(given_path)


