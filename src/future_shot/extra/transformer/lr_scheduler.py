"""
Modules with wrappers for transformers LR Schedulers
"""
import transformers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ConstantWithWarmupLRScheduler(LRScheduler):
    def __new__(
        cls, optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
    ) -> LRScheduler:
        return transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps, last_epoch
        )


class CosineWithWarmupLRScheduler(LRScheduler):
    def __new__(
        cls,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> LRScheduler:
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_cycles, last_epoch
        )


class LinearWithWarmupLRScheduler(LRScheduler):
    def __new__(
        cls,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> LRScheduler:
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch
        )
