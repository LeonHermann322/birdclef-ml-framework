from typing import Any, Literal

from src.models.basic_spectrogram_model import (
    BasicSpectrogramModel,
    BasicSpectrogramModelArgs,
)
from src.datasets.base_dataset import BaseDataset
from src.datasets.birdclef_dataset import BirdClefDataset, BirdClefDatasetArgs
from src.args.yaml_config import YamlConfigModel
from src.optimizers.adam import AdamArgs, create_adam_optimizer
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs


class BirdClefExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, BasicSpectrogramModelArgs, BirdClefDatasetArgs
):
    pass


class BirdClefExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = BirdClefExperimentArgs(**config)
        self.birdclef_data = BirdClefDataset(self.config, yaml_config)
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "birdclef_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.birdclef_data.get_split(split)

    def _create_model(self):
        return BasicSpectrogramModel(self.config)

    @classmethod
    def get_args_model(cls):
        return BirdClefExperimentArgs

    def create_optimizer(self):
        return create_adam_optimizer(self.model, self.config)

    def create_scheduler(self, optimizer):
        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "bce"
