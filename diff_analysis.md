# Codebase vs PixelDiT Paper — Difference Analysis

---

## CRITICAL Issues (likely causing most of the FID gap)

---

### 1. Patch-wise AdaLN instead of Pixel-wise AdaLN [model_jit.py:197, 234-237]

**Code (PixelBlock):**
```python
self.adaLN_proj = nn.Linear(semantic_dim, 6 * pixel_dim, bias=True)  # D → 6*D_pix
# Forward:
sem_mod = self.adaLN_proj(semantic_tokens)             # (N, T, 6*D_pix)
sem_mod = sem_mod.unsqueeze(2).expand(-1, -1, P2, -1)  # (N, T, P², 6*D_pix)  ← BROADCAST!
```

**Paper (Section 3.2, Figure 2-C):**
```python
# Φ: R^D → R^{p² · 6 · D_pix}
self.adaLN_proj = nn.Linear(semantic_dim, patch_size**2 * 6 * pixel_dim)
# Forward:
Θ = Φ(s_cond)  # (B·L, p², 6*D_pix) -- DISTINCT params per pixel, no broadcast
```

The code broadcasts a single projection to all p²=256 pixels in a patch (variant B in paper Figure 2).
PixelDiT uses variant C: a separate set of AdaLN params per pixel via a larger linear layer.

**Impact from paper ablation (Table 4, 80 epochs):**
- Patch-wise AdaLN (variant B): **3.50 FID**
- Pixel-wise AdaLN (variant C): **2.36 FID**  ← the actual design

Fix: change `nn.Linear(semantic_dim, 6 * pixel_dim)` to `nn.Linear(semantic_dim, patch_size**2 * 6 * pixel_dim)` and reshape output directly without broadcast.

---

### 2. Missing Timestep in Pixel-Level Conditioning [model_jit.py:514-515]

**Code:**
```python
for pblock in self.pixel_blocks:
    pixel_tokens = pblock(pixel_tokens, x, hint_mod)  # x = semantic tokens only
```

**Paper (Section 3.1):**
> "we define the conditioning signal for the pixel-level pathway as s_cond := s_N + t, where t is the timestep embedding."

The pixel blocks should receive `x + t_emb.unsqueeze(1)` (semantic tokens + timestep broadcast to all patch positions), not just `x`. Without the timestep, each PiT block has no direct knowledge of the noise level.

Fix: before the pixel block loop, compute `s_cond = x + t_emb.unsqueeze(1)` and pass `s_cond` instead of `x`.

---

### 3. PTC Rate Mismatch [model_jit.py:204-205, main_jit.py:68-69]

**Code:**
```
ptc_rate = 16  # "default 16 for P=16" per comment
# Compresses: 16 pixels → 1 token of dim D_pix=16
# Attention on: T * (P²/r) = 256 * (256/16) = 4096 tokens of dim 16
```

**Paper (Section 3.2):**
> "C: R^{p²×D_pix} → R^D ... reduces the attention sequence length from H×W to L=(H/p)(W/p), a p² fold reduction."

- Compress ALL p²=256 pixels per patch → 1 token of dim D (main hidden dim, e.g. 768)
- Attention on L=256 tokens of dim D=768
- Decompress back to p² pixels of dim D_pix

The code does a 16× reduction (not 256×), uses D_pix=16 for attention (not D_main), resulting in a 4096-token sequence of tiny dim — both less efficient AND less expressive than the paper design.

For correct full compression: set `ptc_rate = patch_size**2 = 256` AND change:
```python
self.ptc_compress   = nn.Linear(patch_size**2 * pixel_dim, hidden_size)
self.ptc_decompress = nn.Linear(hidden_size, patch_size**2 * pixel_dim)
# attention also operates at hidden_size, not pixel_dim
```

---

### 4. Missing REPA Alignment Loss [engine_jit.py, denoiser.py]

**Code:** No REPA. Only diffusion loss.

**Paper (Section 3.4):**
> "L = L_diff + λ_repa * L_repa, λ_repa=0.5, aligned at the 8th patch block using frozen DINOv2 encoder."

REPA encourages mid-level patch tokens to match DINOv2 features. It significantly accelerates convergence and is listed as a core component (without it the model is the Vanilla DiT/16 baseline at 9.84 FID at 80 epochs).

---

## IMPORTANT Issues (contribute meaningfully to worse FID)

---

### 5. Optimizer Beta2 [main_jit.py:199]

**Code:**
```python
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
```

**Paper:** `AdamW, β=(0.9, 0.999)`

β₂=0.95 is a GPT/LLM setting. β₂=0.999 is standard for DiT/vision diffusion models. β₂=0.95 leads to more volatile gradient variance estimates and can hurt convergence stability.

---

### 6. Missing Gradient Clipping [engine_jit.py]

**Code:** No `clip_grad_norm_` in the training loop.

**Paper:** gradient clipping at 1.0 (then 0.5 after LR step-down)

Without clipping, large gradient spikes (especially early in training with the pixel-level pathway initializing) can destabilize training.

---

### 7. Loss Formulation: x0-via-velocity vs Direct Velocity [denoiser.py:86-91]

**Code:**
```python
x_pred = self.net(...)                                          # model predicts x0
v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
loss = ((v - v_pred) ** 2).mean()   # = ((x - x_pred) / (1-t))^2
```
This is x0 prediction with **1/(1-t)² weighting** — strongly upweights near-clean images.

**Paper:** `L = E[||f_θ(x_t, t, y) - v_t||²]` where f_θ **directly predicts v_t** (velocity).

With direct velocity prediction, loss = `||v_pred - (x-ε)||²`, which is uniformly weighted.
The 1/(1-t)² weighting in the code creates instability near t≈1 (even with t_eps=0.05 clamp, max weight = 400×).

---

### 8. Learning Rate [main_jit.py:42, 187-188]

**Code:**
```python
blr = 5e-5  # → lr = blr * eff_batch_size / 256
# With batch=256: lr = 5e-5 × 1.0 = 5e-5
```

**Paper:** lr = 1e-4 constant (for batch=256)

The code's effective LR for a 256-batch run is 2× smaller than PixelDiT.

---

## MINOR / JiT-specific Differences

---

### 9. Timestep Sampling Parameters [denoiser.py:67-69]

**Code:** `P_mean=-0.8, P_std=0.8` (logit-normal biased toward lower t, mean ≈ 0.31)

**Paper:** logit-normal [SD3] — SD3 uses loc=0, scale=1 (mean of sigmoid(0) = 0.5)

P_mean=-0.8 biases training toward noisier images. The SD3 default puts more weight on middle timesteps.

---

### 10. In-context Class Tokens [model_jit.py:343-346, 488-493]

**Code:** Injects 32 in-context class tokens at mid-layer (in_context_start=4 for B, 8 for L).
**Paper:** No such mechanism in PixelDiT.

This is a JiT-specific technique. PixelDiT conditions solely via `c = SiLU(W_t*t + W_y*y + b)`.

---

### 11. GT Pixel Hints — Train/Eval Mismatch [denoiser.py:82-91, model_jit.py:501-507]

**Code:** During training, 25% of pixels get revealed as GT hints (p_hint=0.25) with reduced loss weight (hint_loss_weight=0.1). At inference, hints are all-zero.
**Paper:** No GT hint mechanism in PixelDiT.

The train-eval mismatch (hints present at train, absent at eval) may degrade performance.

---

## Summary Table

| Issue | Severity | Status | Estimated FID Impact |
|-------|----------|--------|----------------------|
| Patch-wise vs pixel-wise AdaLN | CRITICAL | **FIXED** | ~1.1–1.5 FID (paper Table 4) |
| Missing REPA | CRITICAL | **FIXED** | Large (especially early training) |
| Missing timestep in s_cond | CRITICAL | **FIXED** | Unknown, breaks design intent |
| PTC rate wrong (16 vs p²=256) | CRITICAL | **FIXED** | Moderate–large |
| Optimizer β₂ (0.95 vs 0.999) | IMPORTANT | **FIXED** | Moderate |
| Missing gradient clipping | IMPORTANT | **FIXED** | Stability |
| Loss 1/(1-t)² weighting | IMPORTANT | **FIXED** | Moderate |
| LR 2× too low (blr 5e-5 → 1e-4) | IMPORTANT | **FIXED** | Slower convergence |
| P_mean/P_std off (−0.8/0.8 → 0/1) | MINOR | **FIXED** | Small |
| In-context tokens (JiT-specific) | JiT-specific | kept | Unknown |
| GT pixel hints | JiT-specific | kept | Possibly small |
