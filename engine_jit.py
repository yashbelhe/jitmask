import math
import sys
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity


def _swap_to_ema(model_without_ddp):
    """Copy EMA params into model in-place; return saved training param data."""
    params = list(model_without_ddp.parameters())
    saved = [p.data.clone() for p in params]
    for p, ema in zip(params, model_without_ddp.ema_params1):
        p.data.copy_(ema.data)
    return saved


def _swap_to_train(model_without_ddp, saved):
    """Restore training params in-place from saved data."""
    for p, data in zip(model_without_ddp.parameters(), saved):
        p.data.copy_(data)


# ImageNet normalization constants for DINOv2 preprocessing
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def preprocess_for_dino(x_01, target_size=224):
    """
    Preprocess a [0,1] float image tensor for DINOv2.
    x_01: (N, C, H, W) in [0, 1]
    Returns normalized tensor resized to target_size x target_size.
    """
    x = F.interpolate(x_01, size=target_size, mode='bicubic', align_corners=False)
    mean = _DINO_MEAN.to(x.device, x.dtype)
    std  = _DINO_STD.to(x.device, x.dtype)
    return (x - mean) / std


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch,
                    log_writer=None, args=None, dino_encoder=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',        misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
    if dino_encoder is not None:
        metric_logger.add_meter('repa_loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Convert to float [0, 1] — kept for DINOv2 preprocessing
        x_01 = x.to(device, non_blocking=True).to(torch.float32).div_(255.0)

        # Compute frozen DINOv2 features for REPA alignment
        dino_feats = None
        if dino_encoder is not None:
            with torch.no_grad():
                dino_input = preprocess_for_dino(x_01)
                raw = dino_encoder.forward_features(dino_input)
                # timm returns (B, num_patches+1, D); drop CLS token at index 0
                dino_feats = raw[:, 1:] if isinstance(raw, torch.Tensor) else raw['x_norm_patchtokens']
                # dino_feats: (N, T, 768)

        # Normalize to [-1, 1] for diffusion training
        x_norm = x_01 * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, repa_loss_val = model(x_norm, labels, dino_feats=dino_feats)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (PixelDiT: clip at 1.0)
        grad_norm = 0.0
        if args.clip_grad > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item()

        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad_norm=grad_norm)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if repa_loss_val is not None:
            metric_logger.update(repa_loss=repa_loss_val.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('grad_norm',  grad_norm,         epoch_1000x)
                log_writer.add_scalar('lr',         lr,                epoch_1000x)
                if repa_loss_val is not None:
                    log_writer.add_scalar('repa_loss', repa_loss_val.item(), epoch_1000x)


def save_samples(model_without_ddp, args, epoch, fixed_noise, fixed_labels):
    """Save a 3×3 grid of generated samples to disk for visual tracking."""
    if misc.get_rank() != 0:
        return

    model_without_ddp.eval()
    save_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)

    saved = _swap_to_ema(model_without_ddp)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        sampled_images = model_without_ddp.generate(fixed_labels.cuda(), z_init=fixed_noise.cuda())
    _swap_to_train(model_without_ddp, saved)

    # Denormalize [-1,1] → [0,255] uint8
    sampled_images = (sampled_images + 1) / 2
    sampled_images = sampled_images.detach().cpu().numpy()
    sampled_images = np.round(np.clip(sampled_images * 255, 0, 255)).astype(np.uint8)

    _, C, H, W = sampled_images.shape
    grid = np.zeros((H * 3, W * 3, C), dtype=np.uint8)
    for idx in range(9):
        r, c = idx // 3, idx % 3
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = sampled_images[idx].transpose(1, 2, 0)

    cv2.imwrite(os.path.join(save_dir, "epoch_{:04d}.png".format(epoch)), grid[:, :, ::-1])
    print("Saved sample grid to {}/epoch_{:04d}.png".format(save_dir, epoch))


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size  = misc.get_world_size()
    local_rank  = misc.get_rank()
    num_steps   = args.num_images // (batch_size * world_size) + 1

    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1],
            args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Switch to EMA params
    print("Switch to ema")
    saved = _swap_to_ema(model_without_ddp)

    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx   = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    print("Switch back from ema")
    _swap_to_train(model_without_ddp, saved)

    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid             = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid,             epoch)
        log_writer.add_scalar('is{}'.format(postfix),  inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
