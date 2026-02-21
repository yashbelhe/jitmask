import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate, save_samples

from denoiser import Denoiser

torch.set_float32_matmul_precision('high')


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--pixel_hidden_dim', default=16, type=int,
                        help='Hidden dimension for pixel-level blocks (D_pix in PixelDiT)')
    parser.add_argument('--num_pixel_blocks', default=4, type=int,
                        help='Number of pixel transformer blocks (M in PixelDiT). '
                             'PixelDiT: B=2, L=4, XL=4. Default 4 matches L/XL.')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule (constant | cosine | step). '
                             'Use "step" to replicate PixelDiT two-phase training.')
    parser.add_argument('--lr_step_epoch', type=int, default=160,
                        help='Epoch at which to step down the LR (used with --lr_schedule step). '
                             'PixelDiT: 160 for 320-epoch training.')
    parser.add_argument('--lr_after_step', type=float, default=1e-5,
                        help='LR value after step-down (used with --lr_schedule step). '
                             'PixelDiT: 1e-5.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='First EMA decay (used for sampling by default)')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='Second EMA decay')
    # Logit-normal timestep sampling (PixelDiT / SD3 convention: loc=0, scale=1)
    parser.add_argument('--P_mean', default=0.0, type=float,
                        help='Mean of the logit-normal timestep distribution')
    parser.add_argument('--P_std', default=1.0, type=float,
                        help='Std of the logit-normal timestep distribution')
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--p_hint', default=0.25, type=float,
                        help='Fraction of pixels per patch revealed as GT hints during training')
    parser.add_argument('--hint_loss_weight', default=0.1, type=float,
                        help='Loss weight for hinted (revealed) pixels (masked pixels get 1.0)')

    # REPA alignment
    parser.add_argument('--repa_coeff', default=0.5, type=float,
                        help='Weight for REPA alignment loss (0 to disable REPA + DINOv2 entirely)')
    parser.add_argument('--repa_depth', default=8, type=int,
                        help='Patch-level block index at which to extract features for REPA')
    parser.add_argument('--repa_proj_dim', default=768, type=int,
                        help='DINOv2 feature dimension for REPA projector (768 for ViT-B)')
    parser.add_argument('--repa_bf16', action='store_true',
                        help='Compute REPA cosine similarity in bfloat16 (default: float32). '
                             'Use to benchmark whether fp32 REPA is the throughput bottleneck.')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE sampling method (euler | heun)')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Number of ODE sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance scale')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate for evaluation')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def load_dino_encoder(device):
    """Load frozen DINOv2-ViT-B/14 encoder for REPA alignment via timm.
    Uses timm instead of torch.hub to avoid Python 3.10+ union-type syntax
    incompatibility in the cached DINOv2 hub code."""
    import timm
    print("Loading DINOv2-ViT-B/14 encoder for REPA (via timm)...")
    encoder = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0, dynamic_img_size=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)
    encoder = torch.compile(encoder)
    print(f"DINOv2 loaded. Embed dim: {encoder.num_features}")
    return encoder


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Load DINOv2 encoder for REPA (frozen, not wrapped in DDP)
    dino_encoder = None
    if args.repa_coeff > 0:
        dino_encoder = load_dino_encoder(device)

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)
    model = torch.compile(model)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    # PixelDiT: AdamW with betas=(0.9, 0.999)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Fixed noise and labels for consistent sample visualization
    sample_rng = torch.Generator().manual_seed(42)
    fixed_noise = torch.randn(9, 3, args.img_size, args.img_size, generator=sample_rng) * args.noise_scale
    fixed_labels = torch.linspace(0, args.class_num - 1, 9).long()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, model_without_ddp, data_loader_train, optimizer, device, epoch,
            log_writer=log_writer, args=args, dino_encoder=dino_encoder,
        )

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        torch.cuda.empty_cache()
        with torch.no_grad():
            save_samples(model_without_ddp, args, epoch, fixed_noise, fixed_labels)
        torch.cuda.empty_cache()

        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
