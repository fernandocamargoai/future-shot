import os.path
import uuid
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.rank_zero import _warn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import (
    LightningCLI,
    LRSchedulerTypeTuple,
    LRSchedulerTypeUnion,
    ReduceLROnPlateau,
    SaveConfigCallback,
    _global_add_class_path,
    instantiate_class,
)
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.optim import Optimizer

from future_shot.data import FutureShotDataModule
from future_shot.lightning import ClassifierLightningModule


try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class FutureShotLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        default_root_dir = os.path.join("experiments", str(uuid.uuid4()))
        parser.set_defaults({"trainer.default_root_dir": default_root_dir})

        parser.add_argument(
            "--parameter_linking",
            type=Dict[str, str],
            default={},
            help=(
                "Dictionary of parameter names to link to other parameters. "
                "The key is the source parameter and the value is the target parameter. "
                "This allows us to specify parameter linking in the config file. "
            ),
        )

        parser.add_argument(
            "--lr_scheduler_interval",
            type=str,
            default="epoch",
            help=(
                "Interval to call the learning rate scheduler. "
                "Can be 'epoch' or 'step'. "
                "Default: 'epoch'."
            ),
        )

    def _link_parameters(self, subcommand: Optional[str]):
        if subcommand and self.config[subcommand]["parameter_linking"]:
            for source, target in self.config[subcommand]["parameter_linking"].items():
                self.config[subcommand][target] = self.config[subcommand][source]

    def before_instantiate_classes(self) -> None:
        subcommand = self.config.get("subcommand")
        self._link_parameters(subcommand)

        if subcommand == "fit":
            if self.config[subcommand]["trainer"]["logger"]:
                loggers = (
                    self.config[subcommand]["trainer"]["logger"]
                    if isinstance(self.config[subcommand]["trainer"]["logger"], list)
                    else [self.config[subcommand]["trainer"]["logger"]]
                )

                for logger in loggers:
                    log_dir_field = (
                        "log_dir"
                        if "TensorBoardLogger" in logger["class_path"]
                        else "save_dir"
                    )
                    if log_dir_field in logger["init_args"]:
                        logger["init_args"][log_dir_field] = self.config[subcommand][
                            "trainer"
                        ]["default_root_dir"]

                    if "Wandb" in logger["class_path"]:
                        id = os.path.split(
                            self.config[subcommand]["trainer"]["default_root_dir"]
                        )[1]
                        if "id" in logger["init_args"]:
                            logger["init_args"]["id"] = id
                        if "version" in logger["init_args"]:
                            logger["init_args"]["version"] = id
        elif subcommand in ("validate", "test"):
            self.config[subcommand]["trainer"][
                "logger"
            ] = False  # Avoids mistakenly logging to wandb or similar when validating or testing

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRSchedulerTypeUnion] = None,
        lr_scheduler_interval: str = "epoch",
    ) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.module.LightningModule.configure_optimizers`
        method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).
            lr_scheduler_interval: The interval to call the learning rate scheduler. Can be 'epoch' or 'step'.
        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": lr_scheduler.monitor,
                    "interval": lr_scheduler_interval,
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": lr_scheduler_interval,
                },
            }

    def _add_configure_optimizers_method_to_model(
        self, subcommand: Optional[str]
    ) -> None:
        """Overrides the LightningCLI method to add `lr_scheduler_interval`"""
        if not self.auto_configure_optimizers:
            return

        parser = self._parser(subcommand)

        def get_automatic(
            class_type: Union[Type, Tuple[Type, ...]],
            register: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]],
        ) -> List[str]:
            automatic = []
            for key, (base_class, link_to) in register.items():
                if not isinstance(base_class, tuple):
                    base_class = (base_class,)
                if link_to == "AUTOMATIC" and any(
                    issubclass(c, class_type) for c in base_class
                ):
                    automatic.append(key)
            return automatic

        optimizers = get_automatic(Optimizer, parser._optimizers)
        lr_schedulers = get_automatic(LRSchedulerTypeTuple, parser._lr_schedulers)

        if len(optimizers) == 0:
            return

        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(
                f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model` expects at most one optimizer "
                f"and one lr_scheduler to be 'AUTOMATIC', but found {optimizers + lr_schedulers}. In this case the user "
                "is expected to link the argument groups and implement `configure_optimizers`, see "
                "https://lightning.ai/docs/pytorch/stable/common/lightning_cli.html"
                "#optimizers-and-learning-rate-schedulers"
            )

        optimizer_class = parser._optimizers[optimizers[0]][0]
        optimizer_init = self._get(self.config_init, optimizers[0])
        if not isinstance(optimizer_class, tuple):
            optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
        if not optimizer_init:
            # optimizers were registered automatically but not passed by the user
            return

        lr_scheduler_init = None
        if lr_schedulers:
            lr_scheduler_class = parser._lr_schedulers[lr_schedulers[0]][0]
            lr_scheduler_init = self._get(self.config_init, lr_schedulers[0])
            if not isinstance(lr_scheduler_class, tuple):
                lr_scheduler_init = _global_add_class_path(
                    lr_scheduler_class, lr_scheduler_init
                )

        if is_overridden("configure_optimizers", self.model):
            _warn(
                f"`{self.model.__class__.__name__}.configure_optimizers` will be overridden by "
                f"`{self.__class__.__name__}.configure_optimizers`."
            )

        optimizer = instantiate_class(self.model.parameters(), optimizer_init)
        lr_scheduler = (
            instantiate_class(optimizer, lr_scheduler_init)
            if lr_scheduler_init
            else None
        )
        fn = partial(
            self.configure_optimizers,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=self._get(self.config_init, "lr_scheduler_interval"),
        )
        update_wrapper(fn, self.configure_optimizers)  # necessary for `is_overridden`
        # override the existing method
        self.model.configure_optimizers = MethodType(fn, self.model)


class FutureShotSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            super().setup(trainer, pl_module, stage)

            if wandb is not None and wandb.run is not None:
                config_path = os.path.join(trainer.log_dir, self.config_filename)

                artifact = wandb.Artifact(
                    name=f"cli_config-{wandb.run.id}", type="config"
                )
                artifact.add_file(config_path, name="config.yaml")

                wandb.run.log_artifact(artifact)


def cli_main():
    cli = FutureShotLightningCLI(
        ClassifierLightningModule,
        FutureShotDataModule,
        subclass_mode_model=True,
        save_config_callback=FutureShotSaveConfigCallback,
    )
    # note: don't call fit!!
    # TODO: Add custom method to invoke https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.tuner.tuning.Tuner.html


if __name__ == "__main__":
    cli_main()
