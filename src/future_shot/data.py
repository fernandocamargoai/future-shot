import abc
import os
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from datasets import DatasetDict, Split
from datasets.formatting import get_formatter
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
        train_split_key: str = "train",
        validation_split_key: str = "validation",
        test_split_key: str = "test",
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(
            ignore=[
                "dataset",
                "filtering_fn",
                "preprocessing_fn",
                "augmentation_fn",
                "transform_fn",
            ]
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
        self.filtering_fn = filtering_fn
        self.preprocessing_fn = preprocessing_fn
        self.augmentation_fn = augmentation_fn
        self.transform_fn = transform_fn

        self.train_split_key = train_split_key
        self.validation_split_key = validation_split_key
        self.test_split_key = test_split_key

    def _split_train_into_train_and_validation(self) -> None:
        split_dataset = self.dataset[self.train_split_key].train_test_split(
            self.hparams.validation_split,
            seed=self.hparams.seed,
            stratify_by_column=self.hparams.stratify_by_column,
        )
        self.train_dataset = split_dataset[self.train_split_key]
        self.valid_dataset = split_dataset[self.test_split_key]

    def prepare_data(self) -> None:
        if (
            self.test_split_key not in self.dataset
            and self.validation_split_key in self.dataset
        ):
            self.test_dataset = self.dataset[self.validation_split_key]
            self._split_train_into_train_and_validation()
        else:
            self.test_dataset = self.dataset[self.test_split_key]
            if self.validation_split_key not in self.dataset:
                self._split_train_into_train_and_validation()
            else:
                self.train_dataset = self.dataset[self.train_split_key]
                self.valid_dataset = self.dataset[self.validation_split_key]

        if self.filtering_fn is not None:
            self.train_dataset = self.train_dataset.filter(self.filtering_fn)
            self.valid_dataset = self.valid_dataset.filter(self.filtering_fn)
            if self.hparams.apply_filtering_on_test_set:
                self.test_dataset = self.test_dataset.filter(self.filtering_fn)

        if self.preprocessing_fn is not None:
            if self.augmentation_fn is None:
                self.train_dataset = self.train_dataset.map(
                    self.preprocessing_fn, batched=True
                )
            self.valid_dataset = self.valid_dataset.map(
                self.preprocessing_fn, batched=True
            )
            self.test_dataset = self.test_dataset.map(
                self.preprocessing_fn,
                batched=True,
            )

        if self.augmentation_fn is not None:
            self.train_dataset = self.train_dataset.with_transform(self.augmentation_fn)
        else:
            self.train_dataset = self.train_dataset.with_format("torch")

        self.valid_dataset = self.valid_dataset.with_format("torch")
        self.test_dataset = self.test_dataset.with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
