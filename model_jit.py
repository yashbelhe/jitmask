# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# PixelDiT: https://arxiv.org/abs/2511.20645
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Standard single-layer patch embedding: s0 = W_patch · x_patch (PixelDiT Eq. 1)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GeLUFFN(nn.Module):
    """Standard GELU MLP, matching the DiT / PixelDiT patch-level FFN design.
    Replaces the previous SwiGLUFFN which used 2/3× effective width.
    """
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2  = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class PixelBlock(nn.Module):
    """
    Pixel Transformer Block from PixelDiT (https://arxiv.org/abs/2511.20645) Section 3.2.

    Implements:
    - Pixel-wise AdaLN (variant C, Figure 2): Φ: R^D → R^{p²·6·D_pix}, giving DISTINCT
      modulation parameters for each of the p² pixels in a patch (no broadcast).
    - Pixel Token Compaction (PTC): compresses all p² pixel tokens per patch into 1 token
      of dim D (main hidden size), runs self-attention on the short L-token sequence,
      then decompresses back to p² tokens.
    """
    def __init__(self, pixel_dim, hidden_size, num_heads, patch_size):
        super().__init__()
        self.pixel_dim   = pixel_dim
        self.patch_size  = patch_size
        self.num_heads   = num_heads
        self.hidden_size = hidden_size

        P2       = patch_size * patch_size
        head_dim = hidden_size // num_heads

        # Pixel-wise AdaLN: Φ: R^D → R^{P²·6·D_pix}  (distinct per pixel, no broadcast)
        self.adaLN_proj = nn.Linear(hidden_size, P2 * 6 * pixel_dim, bias=True)

        self.norm1 = RMSNorm(pixel_dim)
        self.norm2 = RMSNorm(pixel_dim)

        # PTC: C: R^{P²·D_pix} → R^D  and  E: R^D → R^{P²·D_pix}
        self.ptc_compress   = nn.Linear(P2 * pixel_dim, hidden_size)
        self.ptc_decompress = nn.Linear(hidden_size, P2 * pixel_dim)

        # Self-attention at dim D (sequence length = T = num patches)
        self.qkv       = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.q_norm    = RMSNorm(head_dim)
        self.k_norm    = RMSNorm(head_dim)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)

        # FFN in pixel space
        mlp_hidden = pixel_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(pixel_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, pixel_dim),
        )

    def forward(self, x, s_cond, hint_mod):
        """
        x:        (N, T*P², pixel_dim)  — pixel tokens
        s_cond:   (N, T, hidden_size)   — semantic tokens + timestep (already summed)
        hint_mod: (N, T*P², 6*pixel_dim) — per-pixel hint modulation
        """
        P2 = self.patch_size ** 2
        D  = self.pixel_dim
        H  = self.hidden_size
        N  = x.shape[0]
        T  = s_cond.shape[1]

        # Pixel-wise AdaLN: distinct params for every pixel (no broadcast within patch)
        sem_mod = self.adaLN_proj(s_cond)            # (N, T, P²*6*D)
        sem_mod = sem_mod.reshape(N, T * P2, 6 * D)  # (N, T*P², 6*D)
        mod = sem_mod + hint_mod
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        # === Attention sub-block ===
        h = self.norm1(x) * (1 + scale_a) + shift_a  # (N, T*P², D)

        # PTC compress: all P² pixel tokens → 1 token of dim H
        h = h.reshape(N, T, P2 * D)   # (N, T, P²·D_pix)
        h = self.ptc_compress(h)       # (N, T, H)

        # Self-attention on T tokens at dim H (p²-fold reduction in sequence length)
        B, S, C = h.shape
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, S, C)
        h = self.attn_proj(h)          # (N, T, H)

        # PTC decompress
        h = self.ptc_decompress(h)     # (N, T, P²·D_pix)
        h = h.reshape(N, T * P2, D)   # (N, T*P², D)

        x = x + gate_a * h

        # === FFN sub-block ===
        h = self.norm2(x) * (1 + scale_f) + shift_f
        h = self.ffn(h)
        x = x + gate_f * h

        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = GeLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer with PixelDiT-style pixel-level refinement.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        in_context_len=32,
        in_context_start=8,
        pixel_hidden_dim=16,
        num_pixel_blocks=3,
        repa_depth=8,
        repa_proj_dim=768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.repa_depth = repa_depth

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # single-layer patch projection (PixelDiT: s0 = W_patch · x_patch)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # REPA alignment projector: maps patch-level features → DINOv2 feature space
        self.pixel_hidden_dim = pixel_hidden_dim
        if repa_proj_dim > 0:
            self.repa_projector = nn.Linear(hidden_size, repa_proj_dim)
        else:
            self.repa_projector = None

        # Pixel-level refinement (PixelDiT Section 3.2)
        self.pixel_embed = nn.Linear(in_channels, pixel_hidden_dim)

        # Hint conditioning
        self.hint_mlp = nn.Sequential(
            nn.Linear(in_channels + 1, pixel_hidden_dim),
            nn.SiLU(),
            nn.Linear(pixel_hidden_dim, pixel_hidden_dim),
        )
        self.hint_mod = nn.Linear(pixel_hidden_dim, 6 * pixel_hidden_dim, bias=True)

        self.pixel_blocks = nn.ModuleList([
            PixelBlock(pixel_hidden_dim, hidden_size, num_heads, patch_size)
            for _ in range(num_pixel_blocks)
        ])
        self.pixel_norm_out = RMSNorm(pixel_hidden_dim)
        self.pixel_out = nn.Linear(pixel_hidden_dim, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch projection like nn.Linear (xavier_uniform over flattened weight):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in patch-level blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out hint MLP last layer and hint modulation (hint starts as no-op):
        nn.init.constant_(self.hint_mlp[-1].weight, 0)
        nn.init.constant_(self.hint_mlp[-1].bias, 0)
        nn.init.constant_(self.hint_mod.weight, 0)
        nn.init.constant_(self.hint_mod.bias, 0)

        # Zero-out pixel block adaLN projections (identity at init):
        for block in self.pixel_blocks:
            nn.init.constant_(block.adaLN_proj.weight, 0)
            nn.init.constant_(block.adaLN_proj.bias, 0)

        # Zero-out pixel output layer:
        nn.init.constant_(self.pixel_out.weight, 0)
        nn.init.constant_(self.pixel_out.bias, 0)

    def patchify_to_pixels(self, imgs):
        """
        Patchify image to per-pixel tokens.
        imgs: (N, C, H, W)
        returns: (N, T*P*P, C) where T = (H/P)*(W/P)
        """
        N, C, H, W = imgs.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = imgs.reshape(N, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (N, h, w, p, p, C)
        x = x.reshape(N, h * w * p * p, C)
        return x

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, gt_pixels=None, hint_mask=None, return_repa=False):
        """
        x:          (N, C, H, W) — noised image
        t:          (N,)
        y:          (N,)
        gt_pixels:  (N, T*P*P, C) — GT pixel hints (None at inference)
        hint_mask:  (N, T*P*P, 1) — binary hint mask (None at inference)
        return_repa: if True, return (output, [z_tilde]) for REPA alignment
        """
        x_noised = x  # save for pixel-level blocks

        # class and time embeddings
        t_emb = self.t_embedder(t)   # (N, D)
        y_emb = self.y_embedder(y)   # (N, D)
        c = t_emb + y_emb            # (N, D)

        # patch-level backbone
        x = self.x_embedder(x)
        x += self.pos_embed

        repa_features = None
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

            # Extract intermediate features for REPA at repa_depth
            if i == self.repa_depth and return_repa and self.repa_projector is not None:
                patch_start = self.in_context_len if i >= self.in_context_start else 0
                patch_tokens = x[:, patch_start:]          # (N, T, D)
                repa_features = self.repa_projector(patch_tokens)  # (N, T, repa_proj_dim)

        x = x[:, self.in_context_len:]
        # x: (N, T, hidden_size) — patch semantic tokens s_N

        N, T, _ = x.shape
        P2 = self.patch_size * self.patch_size

        # s_cond = s_N + t  (PixelDiT Section 3.1)
        s_cond = x + t_emb.unsqueeze(1)  # (N, T, D) — timestep added to each patch token

        # GT hint conditioning (all-masked during inference)
        if gt_pixels is None:
            gt_pixels = torch.zeros(N, T * P2, self.in_channels, device=x.device, dtype=x.dtype)
        if hint_mask is None:
            hint_mask = torch.zeros(N, T * P2, 1, device=x.device, dtype=x.dtype)
        hint_input = torch.cat([gt_pixels * hint_mask, hint_mask], dim=-1)  # (N, T*P², C+1)
        hint_cond = self.hint_mlp(hint_input)                   # (N, T*P², D_pix)
        hint_mod_val = self.hint_mod(hint_cond)                 # (N, T*P², 6*D_pix)

        # Embed noised input pixels as pixel tokens
        pixel_tokens = self.patchify_to_pixels(x_noised)    # (N, T*P², C)
        pixel_tokens = self.pixel_embed(pixel_tokens)        # (N, T*P², D_pix)

        # Pixel-level refinement blocks conditioned on s_cond (= s_N + t)
        for pblock in self.pixel_blocks:
            pixel_tokens = pblock(pixel_tokens, s_cond, hint_mod_val)

        # Output projection
        pixel_tokens = self.pixel_norm_out(pixel_tokens)
        pixel_tokens = self.pixel_out(pixel_tokens)                       # (N, T*P², C)
        pixel_tokens = pixel_tokens.reshape(N, T, P2 * self.out_channels) # (N, T, P²*C)
        output = self.unpatchify(pixel_tokens, self.patch_size)

        if return_repa and repa_features is not None:
            return output, [repa_features]
        return output


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               in_context_len=0, in_context_start=0, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               in_context_len=0, in_context_start=0, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               in_context_len=0, in_context_start=0, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               in_context_len=0, in_context_start=0, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               in_context_len=0, in_context_start=0, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               in_context_len=0, in_context_start=0, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
