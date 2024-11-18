r"""
This code is adapted from hyena DNA: https://github.com/HazyResearch/hyena-dna

Sequence Length Warmup by Reloading
====================
Change sequence lengths according to a stage schedule.
The stage parameters sets the sequence length and batch size.
"""

import numpy as np
from lightning.pytorch.callbacks import Callback, GradientAccumulationScheduler
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig


def setup_seqlen_warmup(config: DictConfig):
    trainer_config = config.trainer

    epochs_cume = 0  # track cumulative epochs

    # contains the accumulate_grad_batches schedule to init the trainer
    accumulate_grad_schedule = {}
    trainer_config_dict = dict(trainer_config)
    global_batch_size = trainer_config_dict.pop("global_batch_size")

    for i, stage in enumerate(trainer_config.seqlen_warmup):
        batch_size = stage["batch_size"]  # curr batch size at this stage

        if i == 0:
            config.loader.batch_size = min(batch_size, global_batch_size)

        grad_accum_factor = max(
            1, (global_batch_size // batch_size)
        )  # grad accum factor for this stage
        accumulate_grad_schedule[
            epochs_cume
        ] = grad_accum_factor  # set the grad accum factor for this stage
        epochs_cume += stage["epochs"]  # increment epochs_cume for next stage

    return trainer_config_dict | {
        "strategy": DDPStrategy(
            find_unused_parameters=True, gradient_as_bucket_view=True
        ),
        "callbacks": [
            SeqlenWarmupReload(trainer_config.seqlen_warmup),
            GradientAccumulationScheduler(scheduling=accumulate_grad_schedule),
        ],
    }


class SeqlenWarmupReload(Callback):
    def __init__(self, stage_params: list):
        """
        stage_params is a list of dicts
        e.g. stage_params = [
            {'seq_len': 512, 'epochs': 50},
            {'seq_len': 256, 'epochs': 30},
            {'seq_len': 128, 'epochs': 20},
        ]
        """
        super().__init__()
        assert len(stage_params) > 0, "No stages specified"
        assert all(
            {"seq_len", "epochs"} <= set(stage.keys()) for stage in stage_params
        ), "stage_params must contain keys: seq_len and epochs"

        self.stage_params = stage_params
        self.stage_epochs_cume = np.cumsum([stage["epochs"] for stage in stage_params])

        self._current_stage = 0

    def _verify_stages(self, trainer, model):
        # Double-check that stage parameters are correct,
        # otherwise we'll fail in the middle of training
        for stage in self.stage_params:
            if hasattr(stage, "scheduler"):
                # Verify that we can actually create the scheduler
                # when we need to update it in each stage
                scheduler = model.get_scheduler(trainer.optimizers[0])
                del scheduler

    def on_train_start(self, trainer, model) -> None:
        # Verify all the stage parameters are correct
        self._verify_stages(trainer, model)

        print(f"Training starts at {trainer.current_epoch}")
        if trainer.current_epoch == 0:
            # Update the model to the first stage
            self._update_to_current_stage(trainer, model)
            trainer.reload_dataloaders_every_n_epochs = self.stage_epochs_cume[0]
        else:
            # Preemption or resumption of progressive resizing
            # Update the stage to the current one
            self._current_stage = int(
                np.searchsorted(self.stage_epochs_cume - 1, trainer.current_epoch)
            )
            self._starting_stage = np.any(
                trainer.current_epoch == self.stage_epochs_cume
            )

            print(f"Seq Len Warmup: Restarting at Stage {self._current_stage}")
            if self._starting_stage:
                self._update_lr_scheduler(trainer, model)

            # Set the dataloader and model
            self._update_dataloaders(trainer, model)

            # we don't need to update the model, yet
            # self._update_model(trainer, model)

        return super().on_train_start(trainer, model)

    def _update_lr_scheduler(self, trainer, model):
        if not hasattr(self.stage_params[self._current_stage], "scheduler"):
            # No scheduler specified, so don't update the current scheduler
            return

        assert len(trainer.lr_schedulers) == 1
        # Reinitialize the scheduler
        # We don't need to carry over information from the last scheduler
        # e.g. the last_epoch property, because that will mess with the new
        # scheduler when we step it
        hparams = {
            **model.hparams.scheduler,
            **self.stage_params[self._current_stage]["scheduler"],
        }

        # Note that passing in the optimizer below is okay: the scheduler will be
        # reinitialized and doesn't seem to inherit any current lr info from the
        # optimizer
        trainer.lr_schedulers[0]["scheduler"] = model.get_scheduler(
            trainer.optimizers[0]
        )

        print(f"\tChanged scheduler to {hparams}")

    def _update_dataloaders(self, trainer, model):
        # Set the train resolution and reset the dataloader

        # set new seq len and reset the dataloader
        # max_length should be set in the config of the dataloader
        seq_len = self.stage_params[self._current_stage]["seq_len"]
        # model.hparams.loader.max_length = seq_len

        # we need to resize the batch size too
        batch_size = self.stage_params[self._current_stage].get("batch_size", None)

        # need to change the datamoduke params
        trainer.datamodule.set_curr_seqlen(seq_len)
        trainer.datamodule.set_curr_batch_size(batch_size)

        model.log_dict(
            {
                "callback/epoch": trainer.current_epoch,
                "callback/seq_len": seq_len,
                "callback/batch_size": batch_size,
            },
            sync_dist=True,
        )

        print(
            f"\tAt epoch {trainer.current_epoch}, "
            f"changed Seq Len to {seq_len}, and batch size to {batch_size}"
        )

    def _update_to_current_stage(self, trainer, model):
        print(f"Seq Len Warmup: Moving to Stage {self._current_stage}")
        # Update the train dataloader, model and scheduler
        self._update_dataloaders(trainer, model)
        # self._update_model(trainer, model)
        self._update_lr_scheduler(trainer, model)

    def on_train_epoch_end(self, trainer, model):
        """
        Check to see if new stage is reached for the next epoch, and if so,
        prepare the new stage by changing the dataloader.

        (We do next epoch so that the dataloader is prepared before the next epoch)
        """
        next_epoch = trainer.current_epoch + 1

        # Check if stage should be increased
        if (
            self._current_stage < len(self.stage_params) - 1
            and next_epoch >= self.stage_epochs_cume[self._current_stage]
        ):
            self._current_stage += 1
            self._update_to_current_stage(trainer, model)

        if (
            self._current_stage <= len(self.stage_params) - 1
            and trainer.current_epoch >= self.stage_epochs_cume[self._current_stage - 1]
        ):
            # TODO: keep an eye on lightning documentation as behaviour is not
            # consistent with their naming
            # i.e: should rlly be reload_dataloaders_epoch_interval
            # needs to change after dataloader has been updated from previous stage
            trainer.reload_dataloaders_every_n_epochs = self.stage_params[
                self._current_stage
            ]["epochs"]
            if self._current_stage == len(self.stage_params) - 1:
                trainer.reload_dataloaders_every_n_epochs = 0
                self._current_stage += 1

        return super().on_train_epoch_end(trainer, model)
