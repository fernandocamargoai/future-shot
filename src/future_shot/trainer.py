from pytorch_lightning.cli import LightningCLI

from future_shot.data import FutureShotDataModule
from future_shot.model import FutureShotLightningModule


def cli_main():
    cli = LightningCLI(FutureShotLightningModule, FutureShotDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()