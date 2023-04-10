import abc
import os
from typing import Dict, Any, Optional, Union, List

import datasets
import torch
from datasets import DatasetDict, Split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class FutureShotFiltering(object, metaclass=abc.ABCMeta):

    def setup(self, dataset_info: datasets.DatasetInfo):
        pass

    @abc.abstractmethod
    def __call__(self, example: Dict[str, Any]) -> bool:
        pass


class FutureShotPreprocessing(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
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
        filtering_fn: Optional[FutureShotFiltering] = None,
        apply_filtering_on_test_set: bool = False,
        preprocessing_fn: Optional[FutureShotPreprocessing] = None,
        augmentation_fn: Optional[FutureShotAugmentation] = None,
        transform_fn: Optional[FutureShotTransformation] = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(
            ignore=["dataset", "filtering_fn", "preprocessing_fn", "augmentation_fn", "transform_fn"]
        )

        if isinstance(dataset, DatasetDict):
            self.dataset = dataset
        elif isinstance(dataset, str):
            self.dataset = datasets.load_dataset(dataset)
        else:
            raise ValueError(
                "`dataset` must be either a DatasetDict or a string that will be used to call "
                "`datasets.load_dataset(dataset)`"
            )
        self._filtering_fn = filtering_fn
        self._preprocessing_fn = preprocessing_fn
        self._augmentation_fn = augmentation_fn
        self._transform_fn = transform_fn

    def prepare_data(self) -> None:
        if Split.VALIDATION not in self.dataset:
            split_dataset = self.dataset[Split.TRAIN].train_test_split(
                self.hparams.validation_split,
                seed=self.hparams.seed,
                stratify_by_column=self.hparams.stratify_by_column,
            )
            self._train_dataset = split_dataset[Split.TRAIN]
            self._valid_dataset = split_dataset[Split.TEST]
        else:
            self._train_dataset = self.dataset[Split.TRAIN]
            self._valid_dataset = self.dataset[Split.VALIDATION]
        self._test_dataset = self.dataset[Split.TEST]

        if self._filtering_fn is not None:
            self._train_dataset = self._train_dataset.filter(self._filtering_fn)
            self._valid_dataset = self._valid_dataset.filter(self._filtering_fn)
            if self.hparams.apply_filtering_on_test_set:
                self._test_dataset = self._test_dataset.filter(self._filtering_fn)

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
            self._train_dataset = self._train_dataset.with_transform(
                self._augmentation_fn
            )

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
