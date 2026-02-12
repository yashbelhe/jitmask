# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
    # Use PyTorch's native SDPA which dispatches to FlashAttention-2 when available
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
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PixelBlock(nn.Module):
    """
    Pixel Transformer Block from PixelDiT (https://arxiv.org/abs/2511.20645) Section 3.2.
    Full block: MHSA + FFN, both with pixel-wise adaLN modulation.
    Includes Pixel Token Compaction (PTC) for efficient attention.

    Key design (from paper):
    - γ, β = Linear(S_j)  -- direct linear from patch semantic tokens per block
    - Projection at patch level, then broadcast to all P² pixels per patch
    - PTC: group r consecutive pixel tokens → compress → attention → decompress
    - Zero-init on adaLN for identity mapping at start
    """
    def __init__(self, pixel_dim, semantic_dim, num_heads, patch_size, ptc_rate=16):
        super().__init__()
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.ptc_rate = ptc_rate
        self.num_heads = num_heads
        head_dim = pixel_dim // num_heads

        # Pixel-wise adaLN: 6 params for attn + FFN sub-layers
        # Direct Linear from semantic tokens, projected at patch level
        self.adaLN_proj = nn.Linear(semantic_dim, 6 * pixel_dim, bias=True)

        # Norms for attention and FFN sub-layers
        self.norm1 = RMSNorm(pixel_dim)
        self.norm2 = RMSNorm(pixel_dim)

        # PTC (Pixel Token Compaction, Sec 3.2)
        self.ptc_compress = nn.Linear(ptc_rate * pixel_dim, pixel_dim)
        self.ptc_decompress = nn.Linear(pixel_dim, ptc_rate * pixel_dim)

        # Self-attention (on compacted tokens)
        self.qkv = nn.Linear(pixel_dim, 3 * pixel_dim, bias=True)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attn_proj = nn.Linear(pixel_dim, pixel_dim)

        # FFN
        mlp_hidden = pixel_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(pixel_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, pixel_dim),
        )

    def forward(self, x, semantic_tokens, hint_mod):
        """
        x:                (N, T*P², pixel_dim) - pixel tokens
        semantic_tokens:  (N, T, semantic_dim) - patch-level semantic tokens
        hint_mod:         (N, T*P², 6*pixel_dim) - per-pixel hint modulation
        """
        P2 = self.patch_size ** 2
        r = self.ptc_rate
        D = self.pixel_dim
        N = x.shape[0]
        T = semantic_tokens.shape[1]

        # Pixel-wise adaLN from semantic tokens (project at patch level, broadcast)
        sem_mod = self.adaLN_proj(semantic_tokens)                       # (N, T, 6*D)
        sem_mod = sem_mod.unsqueeze(2).expand(-1, -1, P2, -1)           # (N, T, P², 6*D)
        sem_mod = sem_mod.reshape(N, T * P2, 6 * D)                     # (N, T*P², 6*D)
        mod = sem_mod + hint_mod
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        # === Attention sub-block ===
        h = self.norm1(x) * (1 + scale_a) + shift_a                     # (N, T*P², D)

        # PTC compress: group r consecutive pixel tokens per patch
        h = h.reshape(N, T, P2 // r, r, D)                              # (N, T, P²/r, r, D)
        h = h.reshape(N, T, P2 // r, r * D)                             # (N, T, P²/r, r*D)
        h = self.ptc_compress(h)                                         # (N, T, P²/r, D)
        h = h.reshape(N, T * (P2 // r), D)                              # (N, L_c, D)

        # Multi-head self-attention on compacted tokens
        B, S, C = h.shape
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, S, C)
        h = self.attn_proj(h)

        # PTC decompress
        h = h.reshape(N, T, P2 // r, D)
        h = self.ptc_decompress(h)                                       # (N, T, P²/r, r*D)
        h = h.reshape(N, T, P2 // r, r, D)
        h = h.reshape(N, T * P2, D)                                     # (N, T*P², D)

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
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
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
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        pixel_hidden_dim=16,
        num_pixel_blocks=3,
        pixel_num_heads=4,
        ptc_rate=16,
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

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

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

        # Pixel-level refinement (PixelDiT-style, Sec 3.2)
        self.pixel_hidden_dim = pixel_hidden_dim

        self.pixel_embed = nn.Linear(in_channels, pixel_hidden_dim)

        # Hint conditioning
        self.hint_mlp = nn.Sequential(
            nn.Linear(in_channels + 1, pixel_hidden_dim),
            nn.SiLU(),
            nn.Linear(pixel_hidden_dim, pixel_hidden_dim),
        )
        self.hint_mod = nn.Linear(pixel_hidden_dim, 6 * pixel_hidden_dim, bias=True)

        self.pixel_blocks = nn.ModuleList([
            PixelBlock(pixel_hidden_dim, hidden_size, pixel_num_heads, patch_size, ptc_rate)
            for _ in range(num_pixel_blocks)
        ])
        self.pixel_norm_out = RMSNorm(pixel_hidden_dim)
        self.pixel_out = nn.Linear(pixel_hidden_dim, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out hint MLP last layer and hint modulation (hint starts as no-op):
        nn.init.constant_(self.hint_mlp[-1].weight, 0)
        nn.init.constant_(self.hint_mlp[-1].bias, 0)
        nn.init.constant_(self.hint_mod.weight, 0)
        nn.init.constant_(self.hint_mod.bias, 0)

        # Zero-out pixel block adaLN projections (PixelDiT: each block's semantic projection):
        for block in self.pixel_blocks:
            nn.init.constant_(block.adaLN_proj.weight, 0)
            nn.init.constant_(block.adaLN_proj.bias, 0)

        # Zero-out pixel output layer:
        nn.init.constant_(self.pixel_out.weight, 0)
        nn.init.constant_(self.pixel_out.bias, 0)

    def patchify_to_pixels(self, imgs):
        """
        Patchify image to per-pixel tokens (inverse of unpatchify).
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
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, gt_pixels=None, hint_mask=None):
        """
        x: (N, C, H, W) - noised image
        t: (N,)
        y: (N,)
        gt_pixels: (N, T*P*P, C) - patchified ground truth pixels (optional, training only)
        hint_mask: (N, T*P*P, 1) - binary mask for GT pixel hints (optional, training only)
        """
        # Save noised input for pixel-level blocks (before x_embedder consumes it)
        x_noised = x

        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT patch-level backbone
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]
        # x: (N, T, hidden_size) - patch semantic tokens

        N, T, _ = x.shape
        P2 = self.patch_size * self.patch_size

        # GT hint conditioning (all-masked during inference)
        if gt_pixels is None:
            gt_pixels = torch.zeros(N, T * P2, self.in_channels, device=x.device, dtype=x.dtype)
        if hint_mask is None:
            hint_mask = torch.zeros(N, T * P2, 1, device=x.device, dtype=x.dtype)
        hint_input = torch.cat([gt_pixels * hint_mask, hint_mask], dim=-1)  # (N, T*P², C+1)
        hint_cond = self.hint_mlp(hint_input) * hint_mask                   # (N, T*P², pixel_hidden_dim)
        hint_mod = self.hint_mod(hint_cond) * hint_mask                     # (N, T*P², 6*pixel_hidden_dim)

        # Embed noised input pixels directly as pixel tokens (PixelDiT-style)
        pixel_tokens = self.patchify_to_pixels(x_noised)            # (N, T*P², C)
        pixel_tokens = self.pixel_embed(pixel_tokens)               # (N, T*P², pixel_hidden_dim)

        # Pixel-level refinement blocks with per-block semantic adaLN (PixelDiT Sec 3.2)
        for pblock in self.pixel_blocks:
            pixel_tokens = pblock(pixel_tokens, x, hint_mod)

        # Output projection
        pixel_tokens = self.pixel_norm_out(pixel_tokens)
        pixel_tokens = self.pixel_out(pixel_tokens)                  # (N, T*P², out_channels)
        pixel_tokens = pixel_tokens.reshape(N, T, P2 * self.out_channels)  # (N, T, P²*C)
        output = self.unpatchify(pixel_tokens, self.patch_size)

        return output


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
