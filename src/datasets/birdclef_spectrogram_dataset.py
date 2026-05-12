from typing import Callable, Literal, Optional
from typing_extensions import Self
from pydantic import BaseModel

import torch
import torch.nn.functional as F

from src.datasets.base_dataset import Batch
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, Sample
import librosa
import numpy as np
import pandas as pd
import ast


class BirdClefSpectrogramDatasetArgs(BaseModel):
    # TODO: this is only for debug experiment. However, need to find a good approach to handling long audio inputs
    max_audio_length: int = 4096


class LabelEncoder:
    def __init__(self, taxonomy_file: str) -> None:
        taxonomy = pd.read_csv(taxonomy_file)
        self.class_to_index = {
            label: idx for idx, label in enumerate(taxonomy.primary_label.unique())
        }
        self.index_to_class = {idx: label for label, idx in self.class_to_index.items()}

    def transform_to_label_tensor(self, labels) -> torch.Tensor:
        # Remove duplicates
        labels = list(set(labels))
        label_tensor = torch.zeros(len(self.class_to_index), dtype=torch.long)
        indices = [self.class_to_index[label] for label in labels]
        label_tensor[indices] = 1
        assert label_tensor.sum() == len(
            labels
        ), f"Faulty label tensor: {labels}, expected {len(labels)} positive labels"
        return label_tensor


def load_audio_and_compute_spectrogram(
    file_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    max_length: Optional[int] = None,
):
    # Load audio and compute spectrogram
    if start is None or end is None:
        audio, sr = librosa.load(file_path, sr=32000)
    else:
        audio, sr = librosa.load(
            file_path, sr=32000, offset=start, duration=end - start
        )
    # TODO: mel spectrogram parameter finetuning
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    # Convert to log scale (dB)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if max_length is not None and spectrogram.shape[1] > max_length:
        spectrogram = spectrogram[:, :max_length]

    return spectrogram


class BirdClefSpectrogramSample(Sample):
    def __init__(self, spectrogram, label):
        super().__init__(input=spectrogram, target=label)

    def display(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            self.input.numpy(), sr=32000, x_axis="time", y_axis="mel"
        )
        plt.title(f"Mel spectrogram - Label count: {self.target.sum().item()}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def from_soundscape_label(
        row: pd.Series, label_encoder: LabelEncoder, ds_path: str
    ):
        # Example:
        #       filename	                                start	    end	        primary_label
        #   0   BC2026_Train_0039_S22_20211231_201500.ogg	00:00:00	00:00:05	22961;23158;24321;517063;65380

        filename = f"{ds_path}/{row['filename']}"

        def convert_time_to_seconds(time_str):
            h, m, s = time_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        start = convert_time_to_seconds(row["start"])
        end = convert_time_to_seconds(row["end"])
        labels = row["primary_label"]

        spectrogram = load_audio_and_compute_spectrogram(filename, start, end)

        label_tensor = label_encoder.transform_to_label_tensor(labels.split(";"))
        return BirdClefSpectrogramSample(torch.tensor(spectrogram), label_tensor)

    @staticmethod
    def from_audio_label(row: pd.Series, label_encoder: LabelEncoder, ds_path: str):
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

        spectrogram = load_audio_and_compute_spectrogram(filename)

        label_tensor = label_encoder.transform_to_label_tensor(total_labels)
        return BirdClefSpectrogramSample(torch.tensor(spectrogram), label_tensor)

    @staticmethod
    def from_split_label(
        row: pd.Series, label_encoder: LabelEncoder, max_length: Optional[int] = None
    ):
        spectrogram = load_audio_and_compute_spectrogram(
            row.filename, max_length=max_length
        )
        label_tensor = label_encoder.transform_to_label_tensor(
            ast.literal_eval(row.labels)
        )
        return BirdClefSpectrogramSample(torch.tensor(spectrogram), label_tensor)


class BirdClefSpectrogramDataset(BaseDataset):
    def __init__(
        self, config: BirdClefSpectrogramDatasetArgs, yaml_config: YamlConfigModel
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.label_encoder = LabelEncoder(
            taxonomy_file=yaml_config.base_data_dir + "/taxonomy.csv"
        )

    def get_split(
        self, split: Literal["train"] | Literal["val"] | Literal["test"]
    ) -> Self:
        if split == "train":
            self.items = pd.read_csv(
                self.yaml_config.project_root_dir + "/train_split.csv"
            )
        elif split == "val":
            self.items = pd.read_csv(
                self.yaml_config.project_root_dir + "/val_split.csv"
            )
        elif split == "test":
            self.items = pd.read_csv(
                self.yaml_config.project_root_dir + "/test_split.csv"
            )
        else:
            raise ValueError(f"Invalid split: {split}")
        return self

    def __getitem__(self, index: int) -> BirdClefSpectrogramSample:
        row = self.items.iloc[index]
        return BirdClefSpectrogramSample.from_split_label(
            row, self.label_encoder, max_length=self.config.max_audio_length
        )

    def __len__(self):
        return len(self.items)

    def get_collate_fn(self) -> Callable[[list[Sample]], Batch]:
        def collate_fn(samples: list[Sample]) -> Batch:
            # Input sizes: [128, x] - variable time dimension
            inputs = [sample.input for sample in samples]
            targets = torch.stack([sample.target for sample in samples])

            # Pad all inputs to max length in batch
            max_length = max(inp.shape[1] for inp in inputs)
            padded_inputs = [
                F.pad(inp, (0, max_length - inp.shape[1])) for inp in inputs
            ]
            inputs = torch.stack(padded_inputs)

            return Batch(input=inputs, target=targets)

        return collate_fn
