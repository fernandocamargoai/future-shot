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
    ):
        super().__init__()
        self._model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0 if dense_layer_dim is None else dense_layer_dim
        )
        self._image_field = image_field

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._model(batch[self._image_field])
