import os.path
from typing import Dict, Any, Tuple, Union, List, Optional

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning import LightningModule
from sklearn.metrics import classification_report
from torch import nn
from torchmetrics import Accuracy, MaxMetric, MinMetric

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

    def dot_product(
        self, input_embeddings: torch.Tensor, classes_embeddings: torch.Tensor
    ):
        return input_embeddings @ classes_embeddings.T

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeddings = self.compute_embeddings(features)  # (B, E)

        classes_embeddings = self.normalize(self._class_embedding.weight)  # (N, E)

        x = self.dot_product(input_embeddings, classes_embeddings)  # (B, N)
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
        triplet_loss: nn.Module = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "optimizer", "triplet_loss"])
        # self.save_hyperparameters()

        self._model = FutureShotModel(
            encoder, embedding_dim, num_classes, normalize_embeddings
        )

        self._embeddings_dropout = EmbeddingsDropout(embeddings_dropout)
        if triplet_loss is not None:
            self._triplet_loss = triplet_loss
        else:
            self._triplet_loss = nn.TripletMarginLoss()

        self._train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self._val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        # for logging best so far validation accuracy
        self._val_loss_best = MinMetric()
        self._val_acc_best = MaxMetric()
        self._test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self._test_outputs = []

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model.forward(features)

    def _step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = batch["label"]
        batch_size = labels.size(0)
        anchors = self._model.normalize(self._model.compute_embeddings(batch))  # (B, E)
        positives = self._model.normalize(
            self._model.class_embeddings(labels)
        )  # (B, E)
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

        class_similarities = self._model.dot_product(
            anchors, self._model.normalize(self._model._class_embedding.weight)
        )  # (B, N)
        _, preds = torch.max(class_similarities, 1)  # (B,)

        return loss, preds

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds = self._step(batch)

        self._train_acc(preds, batch["label"])

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self._train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds = self._step(batch)

        self._val_acc(preds, batch["label"])

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self._val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.logged_metrics["val/loss"]
        self._val_loss_best.update(val_loss)

        self.log(
            "val/loss_best",
            self._val_loss_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

        self._val_acc_best.update(self._val_acc.compute())

        self.log(
            "val/acc_best",
            self._val_acc_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds = self._step(batch)

        self._test_acc(preds, batch["label"])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self._test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self._test_outputs.append({"targets": batch["label"], "preds": preds})

        return loss

    def on_test_epoch_end(self) -> None:
        try:
            targets = np.concatenate([output["targets"].detach().cpu() for output in self._test_outputs])
            preds = np.concatenate([output["preds"].detach().cpu() for output in self._test_outputs])

            dataset: Dataset = self.trainer.test_dataloaders.dataset

            report = classification_report(targets, preds, target_names=dataset.features["label"].names, digits=4)

            with open(os.path.join(self.trainer.default_root_dir, "test_classification_report.txt"), "w") as f:
                f.write(report)
        finally:
            self._test_outputs.clear()
