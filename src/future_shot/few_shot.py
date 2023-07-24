import hashlib
import os.path
from glob import glob
from typing import Dict, Any, List, Union, cast

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from jsonargparse import CLI
from pytorch_lightning import Trainer
from sklearn.utils import check_array, check_random_state, indexable
from torch.utils.data import DataLoader
from tqdm import tqdm

from future_shot.data import FutureShotFiltering, FutureShotDataModule
from future_shot.lightning import (
    FutureShotLightningModule,
    FutureShotEmbeddingWrapperLightningModule,
)
from future_shot.trainer import FutureShotLightningCLI


class FilterOutLabelsFiltering(FutureShotFiltering):
    def __init__(self, labels: List[int]):
        self.labels = set(labels)

    def __call__(self, example: Dict[str, Any]) -> bool:
        return example["label"] not in self.labels


class FewShotSplit(object):
    def __init__(
        self,
        n_splits: int = 10,
        train_size: Union[int, float] = None,
        random_state: int = None,
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_indices(self, X, y=None, groups=None):
        y = check_array(y, ensure_2d=False, dtype=None)

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            train = []
            test = []

            for i in range(n_classes):
                if isinstance(self.train_size, int):
                    assert self.train_size < class_counts[i]
                    split_index = self.train_size
                else:
                    split_index = max(round(self.train_size * class_counts[i]), 1)

                shuffled_class_indices = rng.permutation(class_indices[i])
                train.extend(shuffled_class_indices[:split_index])
                test.extend(shuffled_class_indices[split_index:])

            yield train, test

    def split(self, X=None, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test


def _evaluate_few_shot(splitter: FewShotSplit, experiment_dir_paths: List[str]) -> pd.DataFrame:
    metrics = []
    for experiment_dir_path in tqdm(experiment_dir_paths, desc="Evaluating few-shot for each experiment"):
        config_path = os.path.join(experiment_dir_path, "config.yaml")
        checkpoint_path = glob(
            os.path.join(experiment_dir_path, "**", "*.ckpt"), recursive=True
        )[0]

        config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        args = {key: value for key, value in config.items() if key in ("model", "data", "seed_everything")}
        args["trainer"] = {
            "precision": config["trainer"]["precision"],
        }
        cli = FutureShotLightningCLI(
            FutureShotLightningModule,
            FutureShotDataModule,
            args=args,
            subclass_mode_model=True,
            run=False,
            save_config_callback=None,
        )

        data: FutureShotDataModule = cli.datamodule
        model: FutureShotLightningModule = cli.model
        model.load_state_dict(torch.load(checkpoint_path, map_location=model.device)["state_dict"])
        trainer: Trainer = cli.trainer
        trainer.logger = None

        # __main__.FilterOutLabelsFiltering is different from future_shot.few_shot.FilterOutLabelsFiltering,
        # making instance() not work
        assert "FilterOutLabelsFiltering" in str(type(data.filtering_fn))
        filtering_fn: FilterOutLabelsFiltering = cast(
            FilterOutLabelsFiltering, data.filtering_fn
        )
        few_shot_labels = filtering_fn.labels
        data.filtering_fn = None
        data.augmentation_fn = None

        data.prepare_data()

        few_shot_train_dataset = data.train_dataset.filter(
            lambda example: int(example["label"]) in few_shot_labels
        )

        few_shot_train_dataloader = DataLoader(
            dataset=few_shot_train_dataset,
            batch_size=data.hparams.batch_size,
            num_workers=data.hparams.num_workers,
            pin_memory=data.hparams.pin_memory,
            shuffle=False,
        )

        embeddings_batches = trainer.predict(
            FutureShotEmbeddingWrapperLightningModule(model),
            dataloaders=few_shot_train_dataloader,
            return_predictions=True,
        )
        embeddings = torch.cat(embeddings_batches, dim=0)

        labels = few_shot_train_dataset["label"].cpu().detach().numpy()
        for train_indices, _ in tqdm(
            splitter.split(X=None, y=labels, groups=labels), total=splitter.n_splits
        ):
            train_labels = labels[train_indices]

            for few_shot_label in few_shot_labels:
                mask = train_labels == few_shot_label
                new_label_embeddings = embeddings[np.array(train_indices)[mask]]
                model._model._class_embedding.weight.data[few_shot_label] = new_label_embeddings.mean(
                    dim=0
                )

            few_shot_test_dataloader = DataLoader(
                dataset=data.test_dataset,
                batch_size=data.hparams.batch_size,
                num_workers=data.hparams.num_workers,
                pin_memory=data.hparams.pin_memory,
                shuffle=False,
            )

            metrics.append(
                trainer.test(model, dataloaders=few_shot_test_dataloader)
            )

    return pd.DataFrame(data=metrics)


def fit(
    config_path: str,
    num_classes: int,
    num_labels_to_remove: int,
    num_experiments: int,
    seed: int = 42,
    run: bool = False,
):
    random_state = np.random.RandomState(seed)

    # read config file, generate the sha256 hash of the file and use it as the group id
    with open(config_path, "rb") as f:
        group_id = "%s_%d_%d_%d" % (
            hashlib.md5(f.read()).hexdigest(),
            num_labels_to_remove,
            num_experiments,
            seed,
        )

    base_dir = os.path.join("experiments", f"group_{group_id}")

    commands = []

    for i in range(num_experiments):
        labels_to_remove = random_state.choice(
            num_classes, num_labels_to_remove, replace=False
        ).tolist()

        root_dir = os.path.join(base_dir, f"group_{group_id}_experiment_{i}")
        command = f"""WANDB_RUN_GROUP={group_id} \
        python -m future_shot.trainer fit -c {config_path} \
        --data.filtering_fn future_shot.few_shot.FilterOutLabelsFiltering \
        --data.filtering_fn.labels \"{labels_to_remove}\" \
        --trainer.default_root_dir {root_dir}"""

        commands.append(command)

    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "commands.sh"), "w") as f:
        f.write("\n\n".join(commands))

    if run:
        for command in commands:
            os.system(command)


def test(experiments_dir_path: str, n_splits: int = 1000, seed: int = 42):
    experiment_dir_paths = glob(os.path.join(experiments_dir_path, "*"))
    experiment_dir_paths = [
        experiment_dir_path
        for experiment_dir_path in experiment_dir_paths
        if os.path.isdir(experiment_dir_path)
    ]

    for train_size in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1):
        print("Evaluating for train_size=%s" % str(train_size))
        df = _evaluate_few_shot(
            FewShotSplit(
                n_splits=n_splits,
                train_size=train_size,
                random_state=seed,
            ),
            experiment_dir_paths,
        )
        print("Saving results to %s" % os.path.join(experiments_dir_path, f"few_shot_results_{train_size}.csv"))
        df.to_csv(os.path.join(experiments_dir_path, f"few_shot_results_{train_size}.csv"), index=False)


def download_wandb_artifacts(
    wandb_project: str,
    wandb_run_group: str,
):
    group_dir_path = os.path.join("experiments", f"group_{wandb_run_group}")
    os.makedirs(group_dir_path, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(wandb_project, filters={"group": wandb_run_group})

    for run in tqdm(runs, total=runs.length):
        run_dir_path = os.path.join(group_dir_path, run.id)
        os.makedirs(run_dir_path, exist_ok=True)

        for artifact in run.logged_artifacts():
            artifact.download(run_dir_path)


if __name__ == "__main__":
    CLI([fit, test, download_wandb_artifacts])
