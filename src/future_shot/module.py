from typing import Tuple

import torch
from torch import nn


class EmbeddingsDropout(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, *embs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.training and self.p:
            mask = torch.bernoulli(
                torch.tensor(1 - self.p, device=embs[0].device).expand(*embs[0].shape)
            ) / (1 - self.p)

            return tuple(emb * mask for emb in embs)
        return tuple(embs)
