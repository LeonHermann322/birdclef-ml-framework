from typing import Callable, Literal, Optional
from typing_extensions import Self
from pydantic import BaseModel

import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.datasets.base_dataset import Batch
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, Sample
from src.datasets.birdclef_dataset import BirdClefDatasetArgs, BirdClefDataset,BirdClefSample, LabelEncoder
import librosa
import numpy as np
import pandas as pd
import ast

PERCH_SAMPLE_RATE = 32000
PERCH_AUDIO_DURATION_SECONDS = 5
PERCH_AUDIO_LENGTH = PERCH_SAMPLE_RATE * PERCH_AUDIO_DURATION_SECONDS


def time_to_seconds(time_value: object) -> Optional[float]:
    if time_value is None:
        return None
    if isinstance(time_value, float) and np.isnan(time_value):
        return None
    if isinstance(time_value, (int, float, np.integer, np.floating)):
        return float(time_value)

    hours, minutes, seconds = str(time_value).split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


class BirdClefWaveformDatasetArgs(BirdClefDatasetArgs):
    pass


def load_audio_as_waveform(
    file_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    sr: int = PERCH_SAMPLE_RATE,
    target_length: Optional[int] = PERCH_AUDIO_LENGTH,
) -> np.ndarray:
    if start is None or end is None:
        audio, _ = librosa.load(file_path, sr=sr)
    else:
        audio, _ = librosa.load(file_path, sr=sr, offset=start, duration=end - start)

    if target_length is not None:
        if audio.shape[0] > target_length:
            audio = audio[:target_length]
        elif audio.shape[0] < target_length:
            audio = np.pad(audio, (0, target_length - audio.shape[0]))

    return audio


class BirdClefWaveformSample(BirdClefSample):
    def __init__(self, waveform, label):
        super().__init__(input=waveform, target=label)

    def display(self):

        plt.figure(figsize=(10, 4))
        plt.plot(self.input.numpy())
        plt.title(f"Waveform - Label count: {self.target.sum().item()}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def from_soundscape_label(
        row: pd.Series,
        label_encoder: LabelEncoder,
        ds_path: str,
    ):
        # Example:
        #       filename	                                start	    end	        primary_label
        #   0   BC2026_Train_0039_S22_20211231_201500.ogg	00:00:00	00:00:05	22961;23158;24321;517063;65380

        filename = f"{ds_path}/{row['filename']}"

        start = time_to_seconds(row["start"])
        end = time_to_seconds(row["end"])
        labels = row["primary_label"]

        waveform = load_audio_as_waveform(filename, start, end)

        label_tensor = label_encoder.transform_to_label_tensor(labels.split(";"))
        return BirdClefWaveformSample(
            torch.tensor(waveform, dtype=torch.float32), label_tensor
        )

    @staticmethod
    def from_audio_label(
        row: pd.Series,
        label_encoder: LabelEncoder,
        ds_path: str,
    ):
        # Example:
        # primary_label 1161364
        # secondary_labels []
        # type []
        # latitude -22.7562
        # longitude -46.8666
        # scientific_name	Guyalna cuta
        # common_name	Guyalna cuta
        # class_name	Insecta
        # inat_taxon_id	1161364
        # author	Lucas Barbosa
        # license	cc-by-nc
        # rating	0.0
        # url	https://static.inaturalist.org/sounds/1216197....
        # filename	1161364/iNat1216197.ogg
        # collection	iNat

        filename = f"{ds_path}/{row['filename']}"
        primary_label = row["primary_label"]

        def extract_secondary_labels(secondary_labels_str):
            if secondary_labels_str == "[]":
                return []
            return [
                label.replace("'", "").strip()
                for label in secondary_labels_str.strip("[]").split(",")
            ]

        secondary_labels = extract_secondary_labels(row["secondary_labels"])

        total_labels = [primary_label] + secondary_labels

        waveform = load_audio_as_waveform(filename)

        label_tensor = label_encoder.transform_to_label_tensor(total_labels)
        return BirdClefWaveformSample(
            torch.tensor(waveform, dtype=torch.float32), label_tensor
        )

    @staticmethod
    def from_split_label(row: pd.Series, label_encoder: LabelEncoder):
        start = time_to_seconds(row.get("start"))
        end = time_to_seconds(row.get("end"))

        waveform = load_audio_as_waveform(row.filename, start=start, end=end)
        label_tensor = label_encoder.transform_to_label_tensor(
            ast.literal_eval(row.labels)
        )
        return BirdClefWaveformSample(
            torch.tensor(waveform, dtype=torch.float32), label_tensor
        )


class BirdClefWaveformDataset(BirdClefDataset):
    items: pd.DataFrame

    def __init__(
        self, config: BirdClefWaveformDatasetArgs, yaml_config: YamlConfigModel
    ):
        super().__init__(config, yaml_config)
        self.config: BirdClefWaveformDatasetArgs = self.config #for type hinting purpose

    def __getitem__(self, index: int) -> BirdClefWaveformSample:
        row = self.items.iloc[index]
        return BirdClefWaveformSample.from_split_label(row, self.label_encoder)
