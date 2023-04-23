import os.path
import uuid

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback

from future_shot.data import FutureShotDataModule
from future_shot.model import FutureShotLightningModule


try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class FutureShotLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        subcommand = self.config.get("subcommand")
        if subcommand == "fit":
            if self.config[subcommand]["trainer"].get("default_root_dir") is None:
                root_dir = os.path.join("experiments", str(uuid.uuid4()))
                self.config[subcommand]["trainer"]["default_root_dir"] = root_dir
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
                        if (
                            "id" in logger["init_args"]
                        ):
                            logger["init_args"]["id"] = id
                        if (
                            "version" in logger["init_args"]
                        ):
                            logger["init_args"]["version"] = id
        elif subcommand in ("validate", "test"):
            self.config[subcommand]["trainer"][
                "logger"
            ] = False  # Avoids mistakenly logging to wandb or similar when validating or testing


class FutureShotSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            super().setup(trainer, pl_module, stage)

            if wandb is not None and wandb.run is not None:
                config_path = os.path.join(trainer.log_dir, self.config_filename)

                artifact = wandb.Artifact(name=f"cli_config-{wandb.run.id}", type="config")
                artifact.add_file(config_path, name="config.yaml")

                wandb.run.log_artifact(artifact)


def cli_main():
    cli = FutureShotLightningCLI(
        FutureShotLightningModule,
        FutureShotDataModule,
        save_config_callback=FutureShotSaveConfigCallback,
    )
    # note: don't call fit!!
    # TODO: Add custom method to invoke https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.tuner.tuning.Tuner.html


if __name__ == "__main__":
    cli_main()
