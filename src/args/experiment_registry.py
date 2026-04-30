from typing import Type
from src.experiments.birdclef_experiment import BirdClefExperiment
from src.experiments.mnist_experiment import MnistExperiment
from src.experiments.base_experiment import BaseExperiment

experiments: dict[str, Type[BaseExperiment]] = {
    "mnist": MnistExperiment,
    "birdclef": BirdClefExperiment,
}
