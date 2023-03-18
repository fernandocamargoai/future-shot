import abc
import functools
import os
from typing import Callable, Dict, Any, List, Optional, Union

import datasets
import torch
from datasets import DatasetDict, Split, load_dataset
from datasets.formatting import get_formatter
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from future_shot.utils import call_chain


class FutureShotPreprocessing(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass


class FutureShotAugmentation(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass


class FutureShotTransformation(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pass


class FutureShotDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Union[DatasetDict, str],
        batch_size: int = 16,
        pin_memory: bool = True,
        num_workers: int = os.cpu_count(),
        validation_split: Union[float, int] = 0.2,
        stratify_by_column: Optional[str] = None,
        seed: int = 42,
        preprocessing_fn: Optional[FutureShotPreprocessing] = None,
        augmentation_fn: Optional[FutureShotAugmentation] = None,
        transform_fn: Optional[FutureShotTransformation] = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(
            ignore=["dataset", "preprocessing_fn", "augmentation_fn", "transform_fn"]
        )

        if isinstance(dataset, DatasetDict):
            self._dataset = dataset
        elif isinstance(dataset, str):
            self._dataset = datasets.load_dataset(dataset)
        else:
            raise ValueError(
                "`dataset` must be either a DatasetDict or a string that will be used to call "
                "`datasets.load_dataset(dataset)`"
            )
        self._preprocessing_fn = preprocessing_fn
        self._augmentation_fn = augmentation_fn
        self._transform_fn = transform_fn

    def prepare_data(self) -> None:
        if Split.VALIDATION not in self._dataset:
            split_dataset = self._dataset[Split.TRAIN].train_test_split(self.hparams.validation_split, seed=self.hparams.seed, stratify_by_column=self.hparams.stratify_by_column)
            self._train_dataset = split_dataset[Split.TRAIN]
            self._valid_dataset = split_dataset[Split.TEST]
        else:
            self._train_dataset = self._dataset[Split.TRAIN]
            self._valid_dataset = self._dataset[Split.VALIDATION]
        self._test_dataset = self._dataset[Split.TEST]

        if self._preprocessing_fn is not None:
            self._train_dataset = self._train_dataset.map(
                self._preprocessing_fn, batched=True
            )
            self._valid_dataset = self._valid_dataset.map(
                self._preprocessing_fn, batched=True
            )
            self._test_dataset = self._test_dataset.map(
                self._preprocessing_fn, batched=True
            )

        if self._augmentation_fn is not None:
            self._train_dataset = self._train_dataset.with_transform(self._augmentation_fn)

        self._train_dataset = self._train_dataset.with_format("torch")
        self._valid_dataset = self._valid_dataset.with_format("torch")
        self._test_dataset = self._test_dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
