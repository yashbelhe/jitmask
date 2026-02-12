import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            pixel_hidden_dim=args.pixel_hidden_dim,
            num_pixel_blocks=args.num_pixel_blocks,
            pixel_num_heads=args.pixel_num_heads,
            ptc_rate=args.ptc_rate,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # pixel hint conditioning
        self.p_hint = args.p_hint
        self.hint_loss_weight = args.hint_loss_weight

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def patchify_pixels(self, imgs):
        """
        Patchify image into per-pixel tokens (inverse of unpatchify).
        imgs: (N, C, H, W)
        returns: (N, T*P*P, C) where T = (H/P)*(W/P)
        """
        p = self.net.patch_size
        N, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(N, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (N, h, w, p, p, C)
        x = x.reshape(N, h * w * p * p, C)
        return x

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        # Prepare GT pixel hints for conditioning
        N, C, H, W = x.shape
        hint_mask_img = (torch.rand(N, 1, H, W, device=x.device) < self.p_hint).float()  # (N, 1, H, W)
        gt_pixels = self.patchify_pixels(x)                           # (N, T*P², 3)
        hint_mask = self.patchify_pixels(hint_mask_img)               # (N, T*P², 1)

        x_pred = self.net(z, t.flatten(), labels_dropped, gt_pixels=gt_pixels, hint_mask=hint_mask)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # Weighted l2 loss: lower weight for hinted (revealed) pixels
        weight = torch.where(hint_mask_img > 0, self.hint_loss_weight, 1.0)  # (N, 1, H, W)
        loss = ((v - v_pred) ** 2 * weight).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels, z_init=None):
        device = labels.device
        bsz = labels.size(0)
        if z_init is not None:
            z = z_init.to(device)
        else:
            z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
