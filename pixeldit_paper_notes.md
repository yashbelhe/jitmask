# PixelDiT Paper Notes
**Source:** https://arxiv.org/html/2511.20645v1  
**Title:** PixelDiT: Pixel Diffusion Transformers for Image Generation  
**Authors:** Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, Jiebo Luo (NVIDIA / U Rochester)

---

## Key Architecture (Class-conditioned, ImageNet 256×256)

### Model Sizes
| Config       | N (patch) | M (pixel) | D    | D_pix | Heads | Params |
|--------------|-----------|-----------|------|-------|-------|--------|
| PixelDiT-B   | 12        | 2         | 768  | 16    | 12    | 184M   |
| PixelDiT-L   | 22        | 4         | 1024 | 16    | 16    | 569M   |
| PixelDiT-XL  | 26        | 4         | 1152 | 16    | 16    | 797M   |

### Patch-level pathway
- N blocks of augmented DiT (RMSNorm + 2D RoPE replacing LayerNorm/no-RoPE)
- Global conditioning: c = SiLU(W_t * t + W_y * y + b)  -- single vector, broadcast to all L patches
- Standard AdaLN modulation (same parameters for all patch tokens)

### Pixel-level pathway (PiT blocks)
- Input: pixel tokens + s_cond
- **s_cond := s_N + t**  (patch semantic tokens + timestep embedding)  <-- KEY
- M PiT blocks, each with:
  1. Pixel-wise AdaLN  
  2. Pixel Token Compaction (PTC) → attention → decompress
  3. MLP/FFN with pixel-wise AdaLN

### Pixel-wise AdaLN (Variant C in paper, Figure 2)
- Linear projection Φ: R^D → R^{p² · 6 · D_pix}
- Applied per semantic token → gives DISTINCT modulation per pixel
- NOT patch-wise broadcast (variant B) — that's the sub-optimal baseline!
- Output: (β₁, γ₁, α₁, β₂, γ₂, α₂) each of shape (B·L, p², D_pix)

### Pixel Token Compaction (PTC)
- C: R^{p² × D_pix} → R^D  (compress ALL p² pixels per patch → 1 token of dim D_main)
- Self-attention on L = (H/p)*(W/p) tokens of dim D  (p²-fold reduction)
- E: R^D → R^{p² × D_pix}  (decompress)
- For p=16: reduces 65536 → 256 tokens (256x reduction)

---

## Training Details

| Hyperparameter        | PixelDiT Value         |
|-----------------------|------------------------|
| Optimizer             | AdamW β=(0.9, **0.999**) |
| Batch size            | 256                    |
| LR                    | 1e-4 constant → 1e-5   |
| Gradient clipping     | 1.0 → 0.5             |
| EMA decay             | 0.9999                 |
| Timestep sampling     | Logit-normal [SD3], loc=0, scale=1 |
| Mixed precision       | bfloat16              |
| Class drop prob       | 0.1                   |
| Loss                  | Rectified Flow velocity: E[||f_θ(x_t,t,y) - v_t||²] |
| REPA alignment        | λ=0.5, DINOv2 encoder, at 8th patch block |
| Weight decay          | 0                     |
| Patch size            | p=16 (default)        |

### Diffusion formulation
- Rectified Flow: x_t = t·x + (1-t)·ε, velocity v_t = x - ε
- Model **directly predicts v_t** (velocity prediction, NOT x0 prediction)
- Loss: E[||f_θ - v_t||²] with logit-normal timestep weighting

---

## Key Results (PixelDiT-XL)
- 80 epochs: gFID=2.36, CFG=3.25, interval=[0.10,1.00]
- 320 epochs: gFID=1.61, CFG=2.75, interval=[0.10,0.90]

## Ablation (Table 4, PixelDiT-XL, 80 epochs)
- Vanilla DiT/16 (no pixel pathway): 9.84
- + Dual-level + patch-wise AdaLN + PTC: 3.50  ← variant B
- + Pixel-wise AdaLN: 2.36                      ← variant C (the real design)
- Full model at 320 epochs: 1.61
