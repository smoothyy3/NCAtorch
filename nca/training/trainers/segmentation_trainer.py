import torch
from nca.training.trainers.base_trainer import BaseTrainer
from nca.core.losses.loss_functions import DiceBCELoss

class SegmentationTrainer(BaseTrainer):
    """
    Binary segmentation. Channel 0 holds the input image and is frozen across the rollout; 
    channel 1 holds the segmentation logit.
    """

    def _initialize_additional_components(self):
        self.freeze_channels = 1
        if self.loss_fn is None:
            self.loss_fn = DiceBCELoss()
        assert isinstance(self.loss_fn, torch.nn.Module)

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)
        prediction_image, final_state = self.forward(
            initial_state, cond, target,
            logging=logging, freeze_channels=self.freeze_channels,
        )
        logits = prediction_image[:, 1:2]
        loss_dict = self.loss_fn(logits, target)
        return prediction_image, final_state, loss_dict