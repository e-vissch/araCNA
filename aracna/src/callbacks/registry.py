from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from aracna.src.datamodules.simulated.cna_sampler import SimulatedWarmupDifficulty

callback_registry = {
    "warmup_difficulty": SimulatedWarmupDifficulty,
    "model_checkpoint": ModelCheckpoint,
    "learning_rate_monitor": LearningRateMonitor,
}
