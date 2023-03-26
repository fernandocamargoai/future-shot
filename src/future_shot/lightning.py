import abc
from typing import Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score, MaxMetric, MinMetric

from future_shot.model import FutureShotModel, SoftmaxClassifierModel
from future_shot.module import EmbeddingsDropout


class ClassifierLightningModule(LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self._train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self._train_micro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self._train_macro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self._val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self._val_micro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self._val_macro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        # for logging best so far validation accuracy
        self._val_loss_best = MinMetric()
        self._val_acc_best = MaxMetric()
        self._val_micro_f1_best = MaxMetric()
        self._val_macro_f1_best = MaxMetric()
        self._test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self._test_micro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self._test_macro_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self._test_outputs = []

    @abc.abstractmethod
    def _step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds = self._step(batch)

        self._train_acc(preds, batch[self.hparams.label_field])
        self._train_micro_f1(preds, batch[self.hparams.label_field])
        self._train_macro_f1(preds, batch[self.hparams.label_field])

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self._train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/micro_f1",
            self._train_micro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/macro_f1",
            self._train_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds = self._step(batch)

        self._val_acc(preds, batch[self.hparams.label_field])
        self._val_micro_f1(preds, batch[self.hparams.label_field])
        self._val_macro_f1(preds, batch[self.hparams.label_field])

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self._val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/micro_f1",
            self._val_micro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/macro_f1",
            self._val_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.logged_metrics["val/loss"]
        self._val_loss_best.update(val_loss)
        self._val_acc_best.update(self._val_acc.compute())
        self._val_micro_f1_best.update(self._val_micro_f1.compute())
        self._val_macro_f1_best.update(self._val_macro_f1.compute())

        self.log(
            "val/loss_best",
            self._val_loss_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/acc_best",
            self._val_acc_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/micro_f1_best",
            self._val_micro_f1_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/macro_f1_best",
            self._val_macro_f1_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, preds = self._step(batch)

        self._test_acc(preds, batch[self.hparams.label_field])
        self._test_micro_f1(preds, batch[self.hparams.label_field])
        self._test_macro_f1(preds, batch[self.hparams.label_field])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "test/acc", self._test_acc, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/micro_f1",
            self._test_micro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test/macro_f1",
            self._test_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self._test_outputs.append(
            {"targets": batch[self.hparams.label_field], "preds": preds}
        )

        return loss

    def on_test_epoch_end(self) -> List[Dict[str, torch.Tensor]]:
        try:
            return self._test_outputs
            # targets = np.concatenate(
            #     [output["targets"].detach().cpu() for output in self._test_outputs]
            # )
            # preds = np.concatenate(
            #     [output["preds"].detach().cpu() for output in self._test_outputs]
            # )

            # dataset: Dataset = self.trainer.test_dataloaders.dataset
            #
            # report = classification_report(
            #     targets, preds, target_names=dataset.features[self.hparams.label_field].names, digits=4
            # )
            #
            # with open(
            #     os.path.join(
            #         self.trainer.default_root_dir, "test_classification_report.txt"
            #     ),
            #     "w",
            # ) as f:
            #     f.write(report)
        finally:
            self._test_outputs.clear()


class FutureShotLightningModule(ClassifierLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        normalize_embeddings: bool = False,
        embeddings_dropout: float = 0.0,
        triplet_loss: nn.Module = None,
        label_field: str = "label",
    ) -> None:
        super().__init__(num_classes=num_classes)

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

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model.forward(features)

    def compute_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model.compute_embeddings(features)

    def insert_class(self, index: int, embedding: torch.Tensor):
        self._model.insert_class(index, embedding)

    def _step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = batch[self.hparams.label_field]
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
            anchors.reshape(batch_size, 1, self._model.class_embeddings.embedding_dim)
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
            anchors, self._model.normalize(self._model.class_embeddings.weight)
        )  # (B, N)
        _, preds = torch.max(class_similarities, 1)  # (B,)

        return loss, preds


class FutureShotEmbeddingWrapperLightningModule(LightningModule):
    def __init__(self, wrapper_model: FutureShotLightningModule) -> None:
        super().__init__()

        self._wrapped_model = wrapper_model

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._wrapped_model.compute_embeddings(features)


class SoftmaxLightningModule(ClassifierLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        loss: nn.Module = None,
        label_field: str = "label",
    ) -> None:
        super().__init__(num_classes=num_classes)

        self.save_hyperparameters(ignore=["encoder", "optimizer", "loss"])

        self._model = SoftmaxClassifierModel(encoder, embedding_dim, num_classes)

        if loss is not None:
            self._loss = loss
        else:
            self._loss = nn.CrossEntropyLoss()

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model.forward(features)

    def compute_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model.compute_embeddings(features)

    def _step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._model(batch)
        loss = self._loss(logits, batch[self.hparams.label_field])

        return loss, torch.softmax(logits, dim=1)
