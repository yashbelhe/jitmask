import math


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate according to the selected schedule."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.lr_schedule == "step":
            # PixelDiT two-phase schedule: constant lr until lr_step_epoch,
            # then drop to lr_after_step (e.g. 1e-4 → 1e-5 at epoch 160).
            if epoch < args.lr_step_epoch:
                lr = args.lr
            else:
                lr = args.lr_after_step
        else:
            raise NotImplementedError(f"Unknown lr_schedule: {args.lr_schedule}")
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
