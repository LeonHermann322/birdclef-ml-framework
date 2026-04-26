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
        label_tensor = torch.zeros(len(self.class_to_index), dtype=torch.long)
        indices = [self.class_to_index[label] for label in labels]
        label_tensor[indices] = 1
        assert label_tensor.sum() == len(
            labels
        ), "Label tensor should have as many 1s as there are labels"
        return label_tensor


class BirdClefSoundScapeSample(Sample):
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
    def from_label_row(row: pd.Series, label_encoder: LabelEncoder, ds_path: str):
        filename = f"{ds_path}/{row['filename']}"

        def convert_time_to_seconds(time_str):
            h, m, s = time_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        start = convert_time_to_seconds(row["start"])
        end = convert_time_to_seconds(row["end"])
        labels = row["primary_label"]

        # Load audio and compute spectrogram
        audio, sr = librosa.load(filename, sr=32000, offset=start, duration=end - start)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        # Convert to log scale (dB)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        label_tensor = label_encoder.transform_to_label_tensor(labels.split(";"))
        return BirdClefSoundScapeSample(torch.tensor(spectrogram), label_tensor)
