import torch

from src.datasets.base_dataset import Batch
from src.args.yaml_config import YamlConfigModel
from src.models.base_model import BaseModel, Loss, ModelOutput
from perch_hoplite.zoo import model_configs
import os
from torch.nn import BCEWithLogitsLoss
import pandas as pd


class LabelMapping:
    def __init__(self, yaml_config: YamlConfigModel) -> None:
        local_taxonomy_file_path = (
            yaml_config.base_data_dir + "/perch_taxonomy_labels.csv"
        )
        birdclef_taxonomy_file_path = yaml_config.base_data_dir + "/taxonomy.csv"

        if not os.path.exists(local_taxonomy_file_path):
            taxonomy_file_link = (
                "https://huggingface.co/cgeorgiaw/Perch/resolve/main/assets/labels.csv"
            )
            os.system(f"wget {taxonomy_file_link} -O {local_taxonomy_file_path}")

        # Map Perch taxonomy labels to BirdCLEF taxonomy labels
        perch_taxonomy = pd.read_csv(local_taxonomy_file_path).rename(
            columns={"inat2024_fsd50k": "scientific_name"}
        )
        birdclef_taxonomy = pd.read_csv(birdclef_taxonomy_file_path)

        # Calculate exact scientific name matches
        exact_matches = set(perch_taxonomy["scientific_name"]).intersection(
            set(birdclef_taxonomy["scientific_name"])
        )
        print(f"Number of exact matches: {len(exact_matches)}")

        # Mapping from class index in Perch to class index in BirdCLEF
        self.mapping: dict[int, int] = {}

        # Add exact matches to the mapping, index is row index in both csvs
        for scientific_name in exact_matches:
            perch_index = perch_taxonomy[
                perch_taxonomy["scientific_name"] == scientific_name
            ].index[0]
            birdclef_index = birdclef_taxonomy[
                birdclef_taxonomy["scientific_name"] == scientific_name
            ].index[0]
            self.mapping[perch_index] = birdclef_index

        missing_mappings = set(birdclef_taxonomy["scientific_name"]).difference(
            set(perch_taxonomy["scientific_name"])
        )
        print(f"Number of missing mappings: {len(missing_mappings)}")

        # BirdCLEF class indices for which Perch logits are available.
        self.birdclef_indices = sorted(self.mapping.values())

    def map_perch_to_birdclef(self, logits: torch.Tensor) -> torch.Tensor:
        # Create a new tensor with the number of classes in BirdCLEF
        mapped_logits = torch.zeros((logits.shape[0], 234), device=logits.device)
        for perch_index, birdclef_index in self.mapping.items():
            mapped_logits[:, birdclef_index] = logits[:, perch_index]
        return mapped_logits


class PerchModel(BaseModel):
    def __init__(self, yaml_config: YamlConfigModel):
        super().__init__()
        self.label_mapping = LabelMapping(yaml_config)
        self.model = model_configs.load_model_by_name("perch_v2")
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, batch: Batch) -> ModelOutput:
        embed_output = self.model.embed(batch.input.numpy())
        assert embed_output.logits is not None

        return ModelOutput(
            logits=self.label_mapping.map_perch_to_birdclef(
                torch.from_numpy(embed_output.logits["label"].copy())
            )
        )

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> Loss:
        assert batch.target is not None
        assert len(self.label_mapping.birdclef_indices) > 0
        mapped_birdclef_indices = torch.tensor(
            self.label_mapping.birdclef_indices,
            dtype=torch.long,
            device=outputs.logits.device,
        )
        # Calculate the BCE only on BirdCLEF classes represented in the mapping.
        mapped_logits = outputs.logits.index_select(1, mapped_birdclef_indices)
        mapped_targets = batch.target.float().index_select(1, mapped_birdclef_indices)
        loss_value = self.loss_fn(mapped_logits, mapped_targets)
        # TODO: remove sanity check...
        # Multi-label accuracy over mapped BirdCLEF classes only.
        # decision_threshold = 1.0 / mapped_targets.sum()
        # predictions = (torch.softmax(mapped_logits, dim=1) > decision_threshold).float()

        k = int(mapped_targets.sum().item())
        topk_indices = torch.topk(mapped_logits, k=max(k, 1), dim=1).indices
        predictions = torch.zeros_like(mapped_logits).scatter_(1, topk_indices, 1.0)

        accuracy = (predictions == mapped_targets).float().mean()

        tp = (predictions * mapped_targets).sum()
        fp = (predictions * (1 - mapped_targets)).sum()
        tn = ((1 - predictions) * (1 - mapped_targets)).sum()
        fn = ((1 - predictions) * mapped_targets).sum()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        return Loss(
            loss_value,
            {
                "accuracy": accuracy.item(),
                "bce_loss": loss_value.detach().item(),
                "f1": f1.item(),
                "sensitivity": sensitivity.item(),
                "specificity": specificity.item(),
            },
        )
