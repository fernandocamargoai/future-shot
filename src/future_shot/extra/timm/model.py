from typing import Dict, Any, TYPE_CHECKING, List, Union, Optional

import torch
from torch import nn
import timm


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dense_layer_dim: Optional[int] = None,
        image_field: str = "img",
        drop_rate: float = 0.0
    ):
        super().__init__()
        self._model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0 if dense_layer_dim is None else dense_layer_dim
        )
        self._dropout = nn.Dropout(drop_rate)
        self._image_field = image_field

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        features = self._model(batch[self._image_field])
        features = self.drop(features)
        return features
