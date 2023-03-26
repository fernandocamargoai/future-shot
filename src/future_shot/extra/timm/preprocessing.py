from typing import Any, Dict, List, Optional, Tuple

import timm
import timm.data
from datasets.formatting import TorchFormatter
from timm.data import ToTensor
from torchvision.transforms import (
    AutoAugment,
    AutoAugmentPolicy,
    Compose,
    Normalize,
    Resize,
)

from future_shot.data import FutureShotAugmentation, FutureShotPreprocessing


class TimmTransform(Compose):
    def __new__(
        cls,
        model_name: str = "resnet50",
        input_size: Optional[Tuple[int, int, int]] = None,
        is_training: bool = False,
        scale: Tuple[float, float] = (1.0, 1.0),
        ratio: Tuple[float, float] = (1.0, 1.0),
        hflip: float = 0.0,
        vflip: float = 0.0,
        brightness: Tuple[float, float] = (1.0, 1.0),
        contrast: Tuple[float, float] = (1.0, 1.0),
        saturation: Tuple[float, float] = (1.0, 1.0),
        hue: Tuple[float, float] = (0.0, 0.0),
        auto_augment: Optional[str] = None,
    ) -> Compose:
        model = timm.create_model(model_name)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        if input_size is not None:
            data_cfg["input_size"] = tuple(input_size)
        return timm.data.create_transform(
            **data_cfg,
            is_training=is_training,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=(brightness, contrast, saturation, hue),
            auto_augment=auto_augment,
        )


class TimmTransformWithAutoAugmentPolicy(Compose):
    def __new__(
        cls,
        model_name: str = "resnet50",
        input_size: Optional[Tuple[int, int, int]] = None,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
    ) -> Compose:
        model = timm.create_model(model_name)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        if input_size is not None:
            data_cfg["input_size"] = tuple(input_size)

        transform = timm.data.create_transform(**data_cfg)

        transform.transforms.insert(-2, AutoAugment(policy=policy))

        return transform


class TimmFutureShotPreprocessing(FutureShotPreprocessing):
    def __init__(self, image_field: str, transform: Compose) -> None:
        self._image_field = image_field
        self._transform = transform

    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return {
            self._image_field: [
                self._transform(image.convert("RGB"))
                for image in batch[self._image_field]
            ]
        }


class TimmFutureShotAugmentation(FutureShotAugmentation):
    def __init__(self, image_field: str, transform: Compose) -> None:
        self._image_field = image_field
        self._transform = transform
        self._formatter = TorchFormatter()

    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        batch[self._image_field] = [
            self._transform(image.convert("RGB")) for image in batch[self._image_field]
        ]
        batch = self._formatter.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._formatter._consolidate(batch[column_name])
        return batch
