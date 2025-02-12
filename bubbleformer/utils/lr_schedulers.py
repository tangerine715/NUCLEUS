from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


class CosineWarmupLR(SequentialLR):
    """
    Cosine learning rate scheduler with linear warmup
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        warmup_iters (int): Number of warmup iterations
        max_iters (int): Maximum number of iterations
        eta_min (float): Minimum learning rate
    """
    def __init__(self, optimizer, warmup_iters, max_iters, eta_min=0, last_epoch=-1):
        # Initialize the CosineAnnealingLR scheduler with the optimizer and the max_iters
        super().__init__(
            optimizer,
            schedulers=[
                LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: step / warmup_iters
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max_iters,
                    eta_min=eta_min,
                    last_epoch=last_epoch
                ),
            ],
            milestones=[warmup_iters],
            last_epoch=last_epoch,
        )
