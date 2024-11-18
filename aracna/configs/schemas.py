from dataclasses import dataclass, field
from typing import List, Optional, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from aracna.src.utils.constants import MODULE_DEFUALT_LOSS

INTERPOLATED_FROM_PARENT = MISSING


@dataclass
class HyenaFilterConfig:
    emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
    order: int = 16  # width of the implicit MLP
    seq_len: int = INTERPOLATED_FROM_PARENT
    lr: float = 1e-3
    lr_pos_emb: float = 1e-5
    dropout: float = 0.0
    w: int = 1  # frequency of periodic activations
    wd: int = 0  # weight decay of kernel parameters
    bias: bool = True
    normalized: bool = False
    num_inner_mlps: int = 2
    bidirectional: bool = True


@dataclass
class HyenaLayerConfig:
    d_model: int = INTERPOLATED_FROM_PARENT
    l_max: int = 1024
    order: int = 2
    filter_order: int = 64
    num_heads: int = 1
    inner_factor: int = 1
    num_blocks: int = 1
    fused_bias_fc: bool = False
    outer_mixing: bool = False
    dropout: float = 0.0
    filter_dropout: float = 0.0
    post_order_ffn: bool = False
    jit_filter: bool = False
    short_filter_order: int = 3
    activation: str = "id"
    filter: HyenaFilterConfig = field(default_factory=HyenaFilterConfig)


@dataclass
class BaseBackboneConfig:
    name: str  # "attn" | "hyena" | "mamba"
    d_model: int = INTERPOLATED_FROM_PARENT
    n_layer: int = INTERPOLATED_FROM_PARENT


@dataclass
class HyenaConfig(BaseBackboneConfig):
    name: str = "hyena"
    d_model: int = INTERPOLATED_FROM_PARENT
    n_layer: int = INTERPOLATED_FROM_PARENT
    d_inner: int = INTERPOLATED_FROM_PARENT
    layer: HyenaLayerConfig = field(default_factory=HyenaLayerConfig)


@dataclass
class AttnConfig(BaseBackboneConfig):
    name: str = "attn"
    d_inner: int = INTERPOLATED_FROM_PARENT
    layer = None
    # if passing these attn flags, then MHA used
    attn_layer_idx: List[int] | None = None
    attn_cfg: dict | None = None


@dataclass
class MambaConfig(BaseBackboneConfig):
    name: str = "mamba"
    ssm_cfg: dict | None = None


@dataclass
class SimpleEmbeddingConfig:
    name: str = "simple_cna"
    input_dim: int = 2
    embed_dim: int = INTERPOLATED_FROM_PARENT
    positional_feature_size: Optional[int] = None


@dataclass
class RealProfileEmbeddingConfig:
    name: str = "real_cna"
    input_dim: int = 2
    embed_dim: int = INTERPOLATED_FROM_PARENT
    chromosome_dim: int = INTERPOLATED_FROM_PARENT
    token_dim: int = INTERPOLATED_FROM_PARENT
    include_position: bool = True
    include_chromosome: bool = True


@dataclass
class SimpleDecoderConfig:
    name: str = "simple"
    decoder_dim: int = INTERPOLATED_FROM_PARENT


@dataclass
class ModelConfig:
    d_model: int = 32
    n_layer: int = 2
    backbone: dict = field(
        default_factory=dict
    )  # HyeanaConfig | AttnConfig | MambaConfig
    embeddings: dict = field(
        default_factory=dict
    )  # SimpleEmbeddingConfig | RealProfileEmbeddingConfig
    decoder: dict = field(
        default_factory=dict
    )  # SimpleDecoderConfig | GlobalDecoderConfig


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class CosineSchedulerConfig:
    name: str = "cosine"
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class TimmCosineSchedulerConfig:
    name: str = "cosine_timm"
    interval: str = "step"
    t_in_epochs: bool = False
    t_initial: int = 300
    lr_min: float = INTERPOLATED_FROM_PARENT
    warmup_lr_init: float = 1e-6
    warmup_t: int = 10


@dataclass
class TrainerStageConfig:
    epochs: int = INTERPOLATED_FROM_PARENT  # if only one should be trainer.max_epochs
    seq_len: int = 10000
    batch_size: int = 1  # accumaulation is done in the dataloader to global_batch_size


@dataclass
class TrainerConfig:
    devices: Union[int, str] = 1
    accelerator: str = "gpu"
    # accumulate_grad_batches: int = 1  # Gradient accumulation every n batches
    max_epochs: int = 200
    val_check_interval: Union[float, int] = 10  # val every n epochs
    # accelerator: ddp # Automatically set if gpus > 1
    gradient_clip_val: Optional[float] = None
    log_every_n_steps: int = 10
    limit_train_batches: float = (
        1.0  # train on full dataset, can be used to toggle quick run
    )
    limit_val_batches: Union[float, int] = 2

    global_batch_size: int = 50
    seqlen_warmup: Optional[List[TrainerStageConfig]] = field(
        default_factory=lambda: [TrainerStageConfig()]
    )


@dataclass
class LoaderConfig:
    batch_size: int = INTERPOLATED_FROM_PARENT


@dataclass
class WandBConfig:
    project: str = "araCNA-models"
    group: str = ""
    job_type: str = "training"
    mode: str = "online"
    name: Optional[str] = None
    save_dir: str = "."
    id: Optional[str] = None


@dataclass
class RealProfileDataset:
    name: str = "real_profile_sampler"
    max_seq_length: int = 10000
    n_batches: int = 100
    data_path: str = "data/ReleasedData/TCGA_SNP6_hg38"


@dataclass
class GlobalProfileDataset(RealProfileDataset):
    name: str = "global_profile_sampler"
    loss_weights: Optional[List[float]] = field(default_factory=lambda: [0.001, 0.05])


@dataclass
class TaskParams:
    name: str = "train"
    dataset: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    loss: str = MODULE_DEFUALT_LOSS
    loss_kwargs: dict = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)

    model_checkpoint: Optional[str] = None
    use_last_seqlen: bool = False  # only relevant if model_checkpoint is not None


@dataclass
class CheckpointConfig:
    monitor: str = "mse"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True
    # dirpath: str = "checkpoints"
    # filename: str = "mse"
    auto_insert_metric_name: bool = False
    verbose: bool = True


@dataclass
class LRMConfig:
    logging_interval: str = "step"  # Literal["step","epoch"]


@dataclass
class CallbackConfig:
    model_checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    learning_rate_monitor: LRMConfig = field(default_factory=LRMConfig)


def validate_config(config: "Config"):
    if config.task.name == "train":
        # everything else from checkpoint
        assert config.model.embeddings["embed_dim"] == config.model.d_model

        # Annoyingly decoder/backbone has to be dict to be composable
        assert config.model.decoder["decoder_dim"] == config.model.d_model

        assert config.model.backbone["d_model"] == config.model.d_model
        assert config.model.backbone["n_layer"] == config.model.n_layer
        assert config.model.backbone["name"] in ["attn", "hyena", "mamba"]

        if config.model.backbone.get("layer") is not None:
            assert config.model.backbone["name"] == "hyena"

        if config.model.backbone.get("ssm_cfg") is not None:
            assert config.model.backbone["name"] == "mamba"

        if config.model.backbone.get("attn_cfg") is not None:
            assert config.model.backbone["name"] == "attn"

        if config.task.model_checkpoint is None:
            assert config.task.use_last_seqlen is False

        if config.model.backbone["name"] == "mamba":
            d_model = config.model.backbone["d_model"]
            expand = config.model.backbone["ssm_cfg"]["expand"]
            headdim = config.model.backbone["ssm_cfg"]["headdim"]
            assert (d_model * expand / headdim) % 8 == 0  # Mamba2 requires this


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[dict] = field(
        default_factory=dict
    )  # CosineSchedulerConfig | TimmCosineSchedulerConfig
    # scheduler: Optional[CosineSchedulerConfig] = field(default_factory=CosineSchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    task: TaskParams = field(default_factory=TaskParams)

    wandb: Optional[WandBConfig] = field(default_factory=WandBConfig)
    callbacks: Optional[dict] = field(
        default_factory=dict
    )  # asdict(CallbackConfig) | optional others

    def __post_init__(self):
        # validation has to done here because of the way hydra works
        validate_config(self)


def register_configs() -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("len", len)
    cs = ConfigStore.instance()
    cs.store(
        name="base_train_config",
        node=Config,
    )
