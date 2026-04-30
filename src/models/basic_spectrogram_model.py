from pydantic import BaseModel as PDBaseModel

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from src.models.base_model import BaseModel, ModelOutput, Loss
from src.datasets.base_dataset import Batch


class BasicSpectrogramModelArgs(PDBaseModel):
    num_classes: int = 234
    num_channels: int = 32


class BasicSpectrogramModel(BaseModel):
    def __init__(self, config: BasicSpectrogramModelArgs):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        # Single conv layer
        self.conv = nn.Conv2d(1, config.num_channels, kernel_size=(3, 3), padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Single dense layer
        self.fc = nn.Linear(config.num_channels, config.num_classes)

        self.relu = nn.ReLU()
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, batch: Batch) -> ModelOutput:
        # Input: [batch, 128, time]
        x = batch.input.unsqueeze(1)  # [batch, 1, 128, time]

        # Single conv + activation
        x = self.conv(x)
        x = self.relu(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch, num_channels]

        # Single dense layer
        logits = self.fc(x)

        return ModelOutput(logits)

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> Loss:
        assert batch.target is not None

        # Multi-label loss
        loss_value = self.loss_fn(outputs.logits, batch.target.float())

        # Multi-label accuracy: all classes correctly predicted
        predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        accuracy = (predictions == batch.target.float()).all(dim=1).float().mean()

        return Loss(
            loss_value,
            {
                "accuracy": accuracy.item(),
                "bce_loss": loss_value.detach().item(),
            },
        )
