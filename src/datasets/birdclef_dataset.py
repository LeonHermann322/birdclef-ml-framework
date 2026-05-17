# Parent Class for BirdCLEF Datasets
import ast
import os
from typing import Callable, Literal, Optional
from typing_extensions import Self
from pydantic import BaseModel
from src.datasets.base_dataset import BaseDataset, Sample
from src.datasets.base_dataset import Batch
from src.args.yaml_config import YamlConfigModel
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np


class BirdClefDatasetArgs(BaseModel):
    only_soundscapes: bool = True


class BirdClefSample(Sample):

    def __init__(self, input, target):
        super().__init__(input=input, target=target)


def create_random_val_split(
    yaml_config, class_mapping, val_fraction=0.2, random_seed=42
):

    print(
        "val_split.csv not found. Creating val_split.csv by splitting train_split.csv"
    )

    np.random.seed(random_seed)
    train_df = pd.read_csv(yaml_config.base_data_dir + "/train_split.csv")

    # create one hot vector
    X = np.array([i for i in range(len(train_df))])
    y = np.array([[0] * len(class_mapping) for _ in range(len(train_df))])

    labels = train_df["primary_label"].apply(lambda x:ast.literal_eval(x)).tolist()
    for idx, labels in enumerate(labels):
        for label in labels:
            label_idx = class_mapping[label]
            y[idx][label_idx] = 1

    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=random_seed
    )

    train_index, test_index = msss1.split(X, y).__next__()

    val_df = train_df.iloc[test_index].reset_index(drop=True)
    train_df = train_df.iloc[train_index].reset_index(drop=True)

    train_df.to_csv(yaml_config.base_data_dir + "/train_split.csv", index=False)
    val_df.to_csv(yaml_config.base_data_dir + "/val_split.csv", index=False)


def load_splits(yaml_config, class_mapping):
    train_path = os.path.join(yaml_config.base_data_dir, "train_split.csv")
    val_path = os.path.join(yaml_config.base_data_dir, "val_split.csv")
    test_path = os.path.join(yaml_config.base_data_dir, "test_split.csv")

    if not os.path.exists(yaml_config.base_data_dir + "/val_split.csv"):
        create_random_val_split(yaml_config, class_mapping)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # TODO: remove this, after adding the file split handle
    train_df = train_df[
        ~train_df["filename"].str.contains("_train_start_")
    ].reset_index(drop=True)
    val_df = val_df[~val_df["filename"].str.contains("_train_start_")].reset_index(
        drop=True
    )
    test_df = test_df[~test_df["filename"].str.contains("_test_start_")].reset_index(
        drop=True
    )

    return train_df, val_df, test_df


class LabelEncoder:
    def __init__(self, taxonomy_file: str) -> None:
        taxonomy = pd.read_csv(taxonomy_file)
        self.class_to_index = {
            label: idx for idx, label in enumerate(taxonomy.primary_label.unique())
        }
        self.index_to_class = {idx: label for label, idx in self.class_to_index.items()}

    def get_class_to_index_mapping(self):
        return self.class_to_index

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


class BirdClefDataset(BaseDataset):
    def __init__(self, config: BirdClefDatasetArgs, yaml_config: YamlConfigModel):
        self.yaml_config = yaml_config
        self.config = config
        self.label_encoder = LabelEncoder(
            taxonomy_file=yaml_config.base_data_dir + "/taxonomy.csv"
        )
        self.train, self.val, self.test = load_splits(
            yaml_config, self.label_encoder.get_class_to_index_mapping()
        )

    def get_split(
        self, split: Literal["train"] | Literal["val"] | Literal["test"]
    ) -> Self:
        if split == "train":
            self.items = self.train
        elif split == "val":
            self.items = self.val
        elif split == "test":
            self.items = self.test
        else:
            raise ValueError(f"Invalid split: {split}")

        soundscape_mask = self.items["filename"].str.contains("soundscape", na=False)

        if self.config.only_soundscapes:
            # Keep only soundscape recordings
            self.items = self.items[soundscape_mask].reset_index(drop=True)
            # Recalculate mask since indices changed
            soundscape_mask = self.items["filename"].str.contains("soundscape", na=False)

        if soundscape_mask.any():
            # Expand each soundscape recording into 5-second windows using the
            # per-window labels file and match rows by filename suffix only.
            soundscape_labels = pd.read_csv(
                self.yaml_config.project_root_dir + "/data/split_soundscapes_labels.csv"
            )
            soundscape_labels["filename_suffix"] = soundscape_labels["filename"].map(
                os.path.basename
            )
            soundscape_rows = self.items[soundscape_mask].copy()
            soundscape_rows["filename_suffix"] = soundscape_rows["filename"].map(
                os.path.basename
            )
            # Drop the original primary_label since we'll use the per-window labels
            soundscape_rows = soundscape_rows.drop(columns=["primary_label"])
            expanded_soundscapes = soundscape_rows.merge(
                soundscape_labels[["filename_suffix", "start", "end", "primary_label"]],
                on="filename_suffix",
                how="inner",
            )
            expanded_soundscapes["labels"] = expanded_soundscapes["primary_label"].map(
                lambda labels: str(labels.split(";")) if pd.notna(labels) else "[]"
            )
            # Exclude windows with no labels
            expanded_soundscapes = expanded_soundscapes[
                expanded_soundscapes["labels"] != "[]"
            ]
            expanded_soundscapes = expanded_soundscapes.drop(
                columns=["filename_suffix", "primary_label"]
            )
            non_soundscapes = self.items[~soundscape_mask].copy()
            self.items = pd.concat(
                [non_soundscapes, expanded_soundscapes], ignore_index=True
            )

        return self

    def __getitem__(self, index: int) -> BirdClefSample:
        raise NotImplementedError(
            "This method should be overridden in a subclass of BirdClefDataset"
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
