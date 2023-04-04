import shutil

from pytorch_lightning.loggers import WandbLogger


class WandbWithLocalFilesDeletionLogger(WandbLogger):
    """WandbLogger with local files deletion."""

    def finalize(self, status: str) -> None:
        super().finalize(status)

        shutil.rmtree(self.save_dir)
        # TODO: remove and find a better way to do this
