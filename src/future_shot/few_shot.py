import hashlib
import os.path
from typing import Dict, Any, List

import numpy as np
from jsonargparse import CLI

from future_shot.data import FutureShotFiltering


class FilterOutLabelsFiltering(FutureShotFiltering):
    def __init__(self, labels: List[int]):
        self.labels = set(labels)

    def __call__(self, example: Dict[str, Any]) -> bool:
        return example["label"] not in self.labels


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


def test():
    pass


if __name__ == "__main__":
    CLI([fit, test])
