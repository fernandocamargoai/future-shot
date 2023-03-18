from typing import Dict, Any, Tuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from future_shot.module import EmbeddingsDropout


class FutureShotModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        
        self._encoder = encoder
        self._class_embedding = nn.Embedding(num_classes, embedding_dim)
        self._normalize_embeddings = normalize_embeddings

    def normalize(self, embeddings: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return torch.nn.functional.normalize(embeddings, p=2, dim=dim)

    def compute_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self._encoder(features)
        if isinstance(output, dict):
            embeddings = output["sentence_embedding"]  # TODO support customizing
        else:
            embeddings = output
        if self._normalize_embeddings:
            return self.normalize(embeddings)

    @property
    def class_embeddings(self) -> nn.Embedding:
        return self._class_embedding

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeddings = self.compute_embeddings(features)  # (B, E)

        classes_embeddings = self.normalize(self._class_embedding.weight)  # (N, E)

        x = input_embeddings @ classes_embeddings.T  # (B, N)
        # x = torch.softmax(x, dim=1)

        return x


class FutureShotLightningModule(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        normalize_embeddings: bool = False,
        embeddings_dropout: float = 0.0,
        triplet_loss: nn.Module = nn.TripletMarginLoss(),
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "optimizer", "triplet_loss"])

        self._model = FutureShotModel(
            encoder, embedding_dim, num_classes, normalize_embeddings
        )

        self._embeddings_dropout = EmbeddingsDropout(embeddings_dropout)
        self._triplet_loss = triplet_loss

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model.forward(features)

    def _step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        labels = batch["label"]
        batch_size = labels.size(0)
        anchors = self._model.normalize(
            self._model.compute_embeddings(batch)
        )  # (B, E)
        positives = self._model.normalize(self._model.class_embeddings(labels))  # (B, E)
        all_labels = torch.arange(
            0, self.hparams.num_classes, dtype=torch.long, device=self.device
        ).repeat(
            batch_size, 1
        )  # (B, L)
        negative_labels_mask = torch.ones(
            batch_size, self.hparams.num_classes, device=self.device
        ).bool()
        negative_labels_mask[
            torch.arange(0, batch_size, device=self.device), labels
        ] = False
        all_negative_labels = all_labels[negative_labels_mask].reshape(
            batch_size, self.hparams.num_classes - 1
        )
        all_negative_labels_embedding = self._model.normalize(
            self._model.class_embeddings(all_negative_labels), dim=2
        )  # (B, L-1, E)
        similarity_between_archor_and_negatives = (
                anchors.reshape(batch_size, 1, self._model._class_embedding.embedding_dim)
                * all_negative_labels_embedding
        ).sum(
            2
        )  # (B, L-1)
        hardest_negative_labels = torch.argmax(
            similarity_between_archor_and_negatives, dim=1
        )  # (B,)
        negatives = all_negative_labels_embedding[
            torch.arange(0, batch_size, device=self.device), hardest_negative_labels
        ]
        anchors, positives, negatives = self._embeddings_dropout(
            anchors, positives, negatives
        )
        # TODO: check l2_regularization
        loss = self._triplet_loss(anchors, positives, negatives)
        if "sample_weight" in batch:
            loss = (loss * batch["sample_weight"]).mean()
        positive_similarity = (anchors * positives).sum(1)
        negative_similarity = (anchors * negatives).sum(1)
        triplet_acc = (positive_similarity > negative_similarity).sum().float() / anchors.size(0)
        return loss, triplet_acc

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        loss, triplet_acc = self._step(batch)

        self.log("train_loss", loss)
        self.log("train_triplet_acc", triplet_acc, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        loss, triplet_acc = self._step(batch)

        self.log("val_loss", loss)
        self.log("val_triplet_acc", triplet_acc, prog_bar=True)

        return loss
