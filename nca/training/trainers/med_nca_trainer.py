import os
from itertools import chain

import torch.optim as optim

from nca.core.models.model_factory import create_model
from nca.training.trainers.med_nca_rollout import (SCALE_FACTOR, derive_patch_size, med_nca_rollout,)
from nca.training.trainers.segmentation_trainer import SegmentationTrainer
from nca.training.training_utils import create_scheduler, export_model


class MedNCATrainer(SegmentationTrainer):
    """
    Med-NCA: two backbones trained jointly by one optimizer in one step.

    self.ca_model is b1 (the low-resolution backbone) and self.b2 is the full-resolution.
    Both are created before the optimizer, so a single Adam covers them and BaseTrainer's existing backward / step / LR
    logic needs no changes.

    Channel layout is inherited from SegmentationTrainer: channel 0 holds
    the input image and is frozen across the rollout, channel 1 holds the
    segmentation logit. The loss is computed on b2s output only. b1
    receives gradient through the upsample, matching Med-NCAs single loss
    setup (Agent_Multi_NCA.batch_step, src/agents/Agent_Multi_NCA.py).
    """

    def _trainable_parameters(self):
        return chain(self.ca_model.parameters(), self.b2.parameters())

    def _initialize_base_optimizers(self):
        """
        Create b2, then one optimizer over both backbones.
        """
        self.b2 = create_model(
            self.config,
            cond_dim=self.config.COND_DIM or 0,
            img_height=self.config.IM_HEIGHT,
            img_width=self.config.IM_WIDTH,
        )

        self.optimizer = optim.Adam(
            self._trainable_parameters(),
            lr=self.config.TRAINING.LEARNING_RATE,
            betas=self.config.TRAINING.OPTIMIZER_BETAS,
        )
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        print(f"Using LR schedule: {self.config.TRAINING.LR_SCHEDULE_MODE}")

        total = sum(
            p.numel() for p in self._trainable_parameters() if p.requires_grad
        )
        print(f"Med-NCA: 2 backbones, {total:,} trainable parameters total")

    def _initialize_additional_components(self):
        super()._initialize_additional_components()
        self.scale_factor = SCALE_FACTOR
        self.patch_size = derive_patch_size(self.config, self.scale_factor)
        print(
            f"Med-NCA: b1 grid {self.patch_size}x{self.patch_size}, "
            f"b2 crop {self.patch_size}x{self.patch_size} "
            f"of {self.config.DATASET.TARGET_SIZE}x{self.config.DATASET.TARGET_SIZE}"
        )

        self._log_x0 = None
        self._log_target = None

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)

        final_state, target = med_nca_rollout(
            b1=self.ca_model,
            b2=self.b2,
            evolver=self.evolver,
            seed=initial_state,
            iter_n=self.get_iter_range(),
            target=target,
            scale_factor=self.scale_factor,
            patch_size=self.patch_size,
            freeze_channels=self.freeze_channels,
        )

        logits = final_state[:, 1:2]
        loss_dict = self.loss_fn(logits, target)

        # Channel 0 is frozen for the whole b2 rollout, so final_state[:, :1] is exactly the cropped input image.
        self._log_x0 = final_state.detach()
        self._log_target = target.detach()

        return final_state, final_state, loss_dict

    def add_img_logs(self, x0, x, target, cond=None):
        """
        Log the cropped tensors instead of the full-resolution ones.
        """
        if self._log_x0 is None:
            return
        super().add_img_logs(self._log_x0, x, self._log_target, cond)

    def save_model(self, iteration):
        """
        Export both backbones as b1_<iteration>.pt / b2_<iteration>.pt.
        """
        models = (("b1", self.ca_model), ("b2", self.b2))
        for name, model in models:
            model.eval()
            export_model(model, os.path.join(self.output_folder, f"{name}_{iteration}.pt"))
        for _, model in models:
            model.train()
