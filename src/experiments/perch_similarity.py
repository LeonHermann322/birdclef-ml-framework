import wandb
import lancedb
from typing import Any, Literal, cast

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, jaccard_score

from src.models.debug_model import DebugPerchModel
from src.models.basic_spectrogram_model import (
    BasicSpectrogramModel,
    BasicSpectrogramModelArgs,
)
from src.datasets.base_dataset import BaseDataset, Batch
from src.datasets.birdclef_spectrogram_dataset import (
    BirdClefSpectrogramDataset,
    BirdClefSpectrogramDatasetArgs,
)
from src.args.yaml_config import YamlConfigModel
from src.optimizers.adam import AdamArgs, create_adam_optimizer
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs


class BirdClefExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    BasicSpectrogramModelArgs,
    BirdClefSpectrogramDatasetArgs,
):
    pass
    


class PerchSimilarity(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = BirdClefExperimentArgs(**config)
        self.birdclef_data = BirdClefSpectrogramDataset(self.config, yaml_config)
        super().__init__(config, yaml_config)
        self.device = self.get_device()

    def get_name(self) -> str:
        return "perch_similarity"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.birdclef_data.get_split(split)

    def _create_model(self):
        #return BasicSpectrogramModel(self.config)
        return DebugPerchModel(self.yaml_config)

    @classmethod
    def get_args_model(cls):
        return BirdClefExperimentArgs

    def create_optimizer(self):
        return create_adam_optimizer(self.model, self.config)

    def create_scheduler(self, optimizer):
        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "bce"
    

    def create_vector_db(self, model, data_loader):

        db = lancedb.connect("/kaggle/working/audio_vectordb")
        table_name = "audio_snippets"
        table = None

        for i, batch in enumerate(data_loader):
            batch = cast(Batch, batch).to(self.device)
            embedding = model.compute_embedding(batch)
            batch_embeddings = embedding.cpu().detach()
            batch_targets = torch.nonzero(batch.target).cpu().detach()
            data = [
                {"vector": emb.tolist(), "label": lbl.tolist()} 
                for emb, lbl in zip(batch_embeddings, batch_targets)
            ]
            if table is None:
                # Create the table on the first batch
                table = db.create_table(table_name, data=data, mode="overwrite")
            else:
                # Append subsequent batches directly to disk
                table.add(data)
        return db, table
    
    def get_top_k_preds_and_labels(self,model,test_loader,table, k=1): 

        db_predictions = []
        labels = []
        for i, batch in enumerate(test_loader):
            batch = cast(Batch, batch).to(self.device)
            embedding = model.compute_embedding(batch)
            batch_embeddings = embedding.cpu().detach().numpy()

            for emb in batch_embeddings:
                results = table.search(emb.tolist()).limit(k).to_list()
                prediction = [res["label"] for res in results]
                db_predictions.extend(prediction)
                
            labels.extend(batch.target.cpu().detach().numpy().tolist())
        print(f"DB Predictions: {db_predictions[:1]}")
        print(f"True Labels: {labels[:1]}")
        print(f"len(db_predictions): {len(db_predictions)}, len(labels): {len(labels)}")
        
        print(f"len dim 1 (db_predictions): {len(db_predictions[0])}, len(labels): {len(labels[0])}")
        return db_predictions, labels

    def evaluate(self, predicted_labels, true_labels):

        y_true = np.array(true_labels)
        y_pred = torch.nn.functional.one_hot(torch.tensor(predicted_labels), num_classes=y_true.shape[-1]).numpy().max(axis=1)
        print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

        # Good if you care about performance on rare labels.
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        h_loss = hamming_loss(y_true, y_pred)
        j_score = jaccard_score(y_true, y_pred, average='samples')

        metrics = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hamming_loss": h_loss,
            "jaccard_samples": j_score,
            "precision_micro": precision_score(y_true, y_pred, average='micro'),
            "recall_micro": recall_score(y_true, y_pred, average='micro')
        }

        print("\n--- Evaluation Metrics ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        return metrics

    
    def run(self): 
        #perform perch similarity mapping and test cosine similarity

        if self.config.use_wandb:
            wandb.login(key=self.yaml_config.wandb_api_key, relogin=True)

        wandb.init(
            project=self.yaml_config.wandb_project_name,
            entity=self.yaml_config.wandb_entity,
            config=self.raw_config,
            name=self.config.wandb_experiment_name,
            dir=self.yaml_config.cache_dir,
            save_code=True,
            mode="online" if self.config.use_wandb else "disabled",
        )
        if wandb.run is None:
            raise Exception("wandb init failed. wandb.run is None")
        
        with wandb.run:
            data_loader = self._create_dataloader("train")
            val_loader = self._create_dataloader("val")
            model = self._create_model()
            _, table = self.create_vector_db(model, data_loader)

            db_predictions, labels = self.get_top_k_preds_and_labels(model, val_loader, table, k=1)

            #evaluate 
            metrics = self.evaluate(db_predictions, labels)

            wandb.log(metrics)






        