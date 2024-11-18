import torch

from aracna.src.learning.soft_prompts import AracnaSoftPrompt
from aracna.src.models.aracna import Aracna
from aracna.src.task_info import registry as task_info_registry
from aracna.src.utils.config import get_logger, get_object_from_registry, package_model_path
from aracna.src.utils.constants import MODEL_AFFECTED_TASK_INFO_KEYS

log = get_logger(__name__)


def make_task_info_compat(config, model_hparams):
    for key, val in model_hparams.task.info.items():
        if key in MODEL_AFFECTED_TASK_INFO_KEYS:
            config.task.info[key] = val
    if model_hparams.task.info.get("max_tot_cn_arch") is None:
        # backwards compatibility
        config.task.info.max_tot_cn_arch = model_hparams.task.info.max_tot_cn
    if config.task.info.max_tot_cn > config.task.info.max_tot_cn_arch:
        log.warning(
            "max_tot_cn of %s cannot be greater \
            than max_tot_cn_arch of %s. \
            Continuing with max_tot_cn_arch.",
            config.task.info.max_tot_cn, config.task.info.max_tot_cn_arch
        )
        config.task.info.max_tot_cn = config.task.info.max_tot_cn_arch


def use_last_seqlen(config, model_hparams):
    last_seqlen = model_hparams.trainer.seqlen_warmup[-1].seq_len
    if any(val.seq_len < last_seqlen for val in config.trainer.seqlen_warmup[:1]):
        log.warning(
            "use_last_seqlen of %s is set to override first seqlen, \
            but you have specified subsequent seqlens that are less this. \
            Continuing with override.", last_seqlen
        )
    config.trainer.seqlen_warmup[0].seq_len = model_hparams.trainer.seqlen_warmup[
        -1
    ].seq_len


def aracna_from_config(config):
    task_info_cls, _ = get_object_from_registry(
        config.task.info, task_info_registry, init=False
    )  # using cls avoids backwards comptibility issues if other
    # inputs to task info changed

    # TODO- only have one decoder atm, maybe just make part of Aracna?
    decoder = task_info_cls.decoder_cls(**config.model.decoder)
    return Aracna(config.model, decoder)


def get_normal_model_statedict(model_hparams, trained_statedict, _):
    new_model = aracna_from_config(model_hparams)
    model_state_dict = {
        k.removeprefix("model."): v for k, v in trained_statedict.items()
    }
    return new_model, model_state_dict


def compatible_model_from_checkpoint(config, get_model=get_normal_model_statedict):
    # only wantget aracna, do not want to recreate whole AracnaTrain
    # TODO-is there a way to change this such that it doesn't have this chainging
    # then we can use lightning load_from_checkpoint?
    model_path = package_model_path(config.task.model_checkpoint)

    trained_model_info = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    
    # lightning will put module to device later, so always construct on cpu
    model_hparams = trained_model_info["hyper_parameters"]

    # backwards compatibility, replace with absolute and not relative defn
    model_hparams.model.decoder.max_tot_cn = model_hparams.task.info.get(
        "max_tot_cn_arch", model_hparams.task.info.max_tot_cn
    )

    model, model_state_dict = get_model(
        model_hparams, trained_model_info["state_dict"], config
    )

    # Load the state dict into the model
    model.load_state_dict(model_state_dict, strict=False)
    
    config.model = model_hparams.model

    make_task_info_compat(config, model_hparams)

    if config.task.get("use_last_seqlen"):
        use_last_seqlen(config, model_hparams)

    return model


def softprompt_from_config(config):
    model = compatible_model_from_checkpoint(config)
    return AracnaSoftPrompt(model, config)


registry = {
    "train": aracna_from_config,
    "softprompt": softprompt_from_config,
    "pretrained": compatible_model_from_checkpoint,
}


scheduler_registry = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}
