import shutil

from pytorch_lightning import Callback, Trainer, LightningModule


class DeleteRootDirOnTeardownCallback(Callback):
    """Delete the root directory on teardown. Meant to be used if you have another callback that uploads necessary
    files to a remote location, such as a WandbLogger, and you want to delete the local files after the run is
    complete."""
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if stage == "fit":
            if trainer.default_root_dir is not None:
                shutil.rmtree(trainer.default_root_dir)
