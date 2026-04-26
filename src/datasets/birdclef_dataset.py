from typing import Optional

import torch

from src.datasets.base_dataset import Sample
import librosa
import numpy as np
import pandas as pd


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
    file_path: str, start: Optional[float] = None, end: Optional[float] = None
):
    # Load audio and compute spectrogram
    if start is None or end is None:
        audio, sr = librosa.load(file_path, sr=32000)
    else:
        audio, sr = librosa.load(
            file_path, sr=32000, offset=start, duration=end - start
        )
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    # Convert to log scale (dB)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram


class BirdClefSample(Sample):
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
        return BirdClefSample(torch.tensor(spectrogram), label_tensor)

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
        return BirdClefSample(torch.tensor(spectrogram), label_tensor)
