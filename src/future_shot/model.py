from typing import Dict

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


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
        self._normalize_embeddings = normalize_embeddings
        self._class_embeddings = nn.Embedding(num_classes, embedding_dim)

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
        return self._class_embeddings

    def insert_class(self, index: int, embedding: torch.Tensor):
        self._class_embeddings.weight.data[index] = embedding.to(
            self._class_embeddings.weight.device
        )

    def dot_product(
        self, input_embeddings: torch.Tensor, classes_embeddings: torch.Tensor
    ):
        return input_embeddings @ classes_embeddings.T

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeddings = self.compute_embeddings(features)  # (B, E)

        classes_embeddings = self.normalize(self._class_embeddings.weight)  # (N, E)

        x = self.dot_product(input_embeddings, classes_embeddings)  # (B, N)
        # x = torch.softmax(x, dim=1)

        return x


class SoftmaxClassifierModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self._encoder = encoder
        self._classifier = nn.Linear(embedding_dim, num_classes)

    def compute_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self._encoder(features)
        if isinstance(output, dict):
            embeddings = output["sentence_embedding"]  # TODO support customizing
        else:
            embeddings = output
        return embeddings

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_embeddings = self.compute_embeddings(features)

        return self._classifier(input_embeddings)
