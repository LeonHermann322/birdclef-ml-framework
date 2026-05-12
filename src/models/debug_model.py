#dummy model to create random output fast 
import torch
from src.args.yaml_config import YamlConfigModel
from src.models.perch_model import PerchModel, LabelMapping
from torch.nn import BCEWithLogitsLoss
from src.datasets.base_dataset import Batch
from src.models.base_model import BaseModel, ModelOutput, Loss

class DebugModel(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, batch: Batch) -> ModelOutput:
        batch_size = batch.input.shape[0]
        num_birdclef_classes = 234
        random_logits = torch.randn(batch_size, num_birdclef_classes, device=batch.input.device)
        return ModelOutput(logits=random_logits)

    def embed(self, input: torch.Tensor) -> ModelOutput:
        batch_size = input.shape[0]
        embedding_dim = 128
        random_embedding = torch.randn(batch_size, embedding_dim, device=input.device)
        return ModelOutput(logits=random_embedding)
    
    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> Loss:
        assert batch.target is not None
        loss_fn = BCEWithLogitsLoss()
        loss_value = loss_fn(outputs.logits, batch.target.float())
        return Loss(loss_value, None)


class DebugPerchModel(PerchModel):
    def __init__(self, yaml_config: YamlConfigModel):
        #bypass perch loading 
        super(PerchModel, self).__init__()
        self.model = DebugModel()
        self.label_mapping = LabelMapping(yaml_config)
        self.loss_fn = BCEWithLogitsLoss()
