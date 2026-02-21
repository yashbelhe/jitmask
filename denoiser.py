import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            pixel_hidden_dim=args.pixel_hidden_dim,
            num_pixel_blocks=args.num_pixel_blocks,
            repa_depth=args.repa_depth,
            repa_proj_dim=args.repa_proj_dim,
        )
        self.img_size    = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean      = args.P_mean
        self.P_std       = args.P_std
        self.t_eps       = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1  = args.ema_decay1
        self.ema_decay2  = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # pixel hint conditioning
        self.p_hint           = args.p_hint
        self.hint_loss_weight = args.hint_loss_weight

        # REPA
        self.repa_coeff = args.repa_coeff

        # generation hyper params
        self.method       = args.sampling_method
        self.steps        = args.num_sampling_steps
        self.cfg_scale    = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def patchify_pixels(self, imgs):
        """
        Patchify image into per-pixel tokens.
        imgs: (N, C, H, W)
        returns: (N, T*P*P, C)
        """
        p = self.net.patch_size
        N, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(N, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (N, h, w, p, p, C)
        x = x.reshape(N, h * w * p * p, C)
        return x

    def sample_t(self, n: int, device=None):
        # Logit-normal timestep sampling (SD3 / PixelDiT convention)
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, dino_feats=None):
        """
        x:          (N, C, H, W) — clean image in [-1, 1]
        labels:     (N,)
        dino_feats: (N, T, D_dino) — frozen DINOv2 patch features (None disables REPA)
        """
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t   = self.sample_t(x.size(0), device=x.device)  # (N,)
        t4d = t.view(-1, 1, 1, 1)
        e   = torch.randn_like(x) * self.noise_scale

        # Rectified flow interpolation: z_t = t*x + (1-t)*noise
        z = t4d * x + (1 - t4d) * e

        # True velocity (direct, PixelDiT Eq. 8)
        v = x - e  # (N, C, H, W)

        # Prepare GT pixel hints for conditioning
        N, C, H, W = x.shape
        hint_mask_img = (torch.rand(N, 1, H, W, device=x.device) < self.p_hint).float()
        gt_pixels = self.patchify_pixels(x)              # (N, T*P², 3)
        hint_mask = self.patchify_pixels(hint_mask_img)  # (N, T*P², 1)

        use_repa = (self.repa_coeff > 0 and dino_feats is not None and self.training)

        net_out = self.net(
            z, t, labels_dropped,
            gt_pixels=gt_pixels, hint_mask=hint_mask,
            return_repa=use_repa,
        )

        if use_repa:
            v_pred, zs_tilde = net_out
        else:
            v_pred = net_out

        # Diffusion loss: direct velocity matching (PixelDiT Eq. 8)
        weight    = torch.where(hint_mask_img > 0, self.hint_loss_weight, 1.0)
        diff_loss = ((v - v_pred) ** 2 * weight).mean()

        if not use_repa:
            return diff_loss, None

        # REPA alignment loss: cosine similarity between model mid-features and DINOv2
        # Cast to float32 by default for numerical stability; use --repa_bf16 to stay in bf16.
        repa_loss = 0.0
        for z_tilde in zs_tilde:
            if getattr(self, 'repa_bf16', False):
                z_tilde_n = F.normalize(z_tilde, dim=-1)
                z_dino_n  = F.normalize(dino_feats, dim=-1)
            else:
                z_tilde_n = F.normalize(z_tilde.float(), dim=-1)
                z_dino_n  = F.normalize(dino_feats.float(), dim=-1)
            repa_loss += -(z_tilde_n * z_dino_n).sum(dim=-1).mean()

        return diff_loss + self.repa_coeff * repa_loss, repa_loss

    @torch.no_grad()
    def generate(self, labels, z_init=None):
        device = labels.device
        bsz    = labels.size(0)
        z = (z_init.to(device) if z_init is not None
             else self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device))

        timesteps = (torch.linspace(0.0, 1.0, self.steps + 1, device=device)
                     .view(-1, *([1] * z.ndim))
                     .expand(-1, bsz, -1, -1, -1))

        stepper = self._euler_step if self.method == "euler" else (
                  self._heun_step  if self.method == "heun"  else None)
        if stepper is None:
            raise NotImplementedError(f"Unknown sampling method: {self.method}")

        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels)
        # last step always Euler for stability
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        """Model forward with CFG. Model now directly predicts velocity."""
        # conditional
        v_cond = self.net(z, t.flatten(), labels)

        # unconditional
        v_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))

        # CFG interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t      = self._forward_sample(z, t, labels)
        z_next_euler  = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        v_pred        = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
