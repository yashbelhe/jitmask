# Additional PixelDiT vs Current Code Differences (Sonnet)

This file documents paper-vs-code differences identified beyond `diff_analysis.md`,
excluding sampling/inference-only settings. Items already covered in `diff_analysis_codex.md`
are noted as such to avoid duplication.

---

## 1. Missing LR Step-Down Schedule [main_jit.py:45, util/lr_sched.py]

*(Also in `diff_analysis_codex.md` #2, included here for completeness.)*

`diff_analysis.md` issue #8 only flags that the base LR is 2× too low. The paper additionally
uses a two-phase schedule:

> "Constant learning rate of 1×10⁻⁴ for the first 160 epochs, then step down to 1×10⁻⁵
> for the remainder."  — `references/pixeldit_2511.20645v1.txt:1398–1411`

The code defaults to `lr_schedule='constant'` with no step-down logic at all. The step-down
is what allows the model to anneal into its best quality after bulk training — missing it means
the model never goes through that final refinement phase regardless of training duration.

**Fix:** implement a step-down scheduler that halves-decades the LR at epoch 160.

---

## 2. Model Depth Mismatch [model_jit.py:527–541, main_jit.py:64–65]

*(Also in `diff_analysis_codex.md` #1.)*

Paper configs (confirmed from Table 6, `references/pixeldit_2511.20645v1.txt:2167–2178`):

| Model       | Patch depth N | Pixel depth M |
|-------------|---------------|---------------|
| PixelDiT-B  | 12            | 2             |
| PixelDiT-L  | 22            | 4             |
| PixelDiT-XL | 26            | 4             |

Code:
- `JiT_L_16` uses `depth=24` (two extra patch blocks vs paper's L).
- Default `--num_pixel_blocks=3` matches neither B nor L.

Neither JiT-B (3 pixel blocks) nor JiT-L (24 patch + 3 pixel blocks) is a direct reproduction
of any paper configuration, making FID comparisons inexact.

---

## 3. SwiGLU FFN in Patch Blocks vs Paper's Standard GELU MLP [model_jit.py:133–151, 283]

**Not in any existing MD file.**

The paper describes the patch-level pathway as "augmented DiT (RMSNorm + 2D RoPE replacing
LayerNorm / no-RoPE)" — the augmentations are explicitly only those two. Standard DiT uses a
plain GELU MLP with `mlp_ratio=4`.

The code uses `SwiGLUFFN` for patch blocks:

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, ...):
        hidden_dim = int(hidden_dim * 2 / 3)   # ← effective width is smaller
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3  = nn.Linear(hidden_dim, dim)
```

With `mlp_ratio=4`, SwiGLU's effective intermediate dim is `4 * 2/3 ≈ 2.67×` the input, vs
GELU's `4×`. This reduces the patch FFN parameter count by ~33% relative to the paper's design.
The pixel blocks use a standard GELU MLP, so the inconsistency is only at the patch level.

SwiGLU is a JiT-specific addition inherited from LightningDiT. It may help or hurt relative
to the paper baseline, but it is an unacknowledged deviation.

---

## 4. Bottleneck Patch Embedding [model_jit.py:29–30]

*(Also in `diff_analysis_codex.md` #4.)*

The paper uses a standard single-layer patch embedding `s₀ = W_patch · x_patch`. The code
uses a two-stage bottleneck:

```python
self.proj1 = nn.Conv2d(in_chans, pca_dim=128, kernel_size=p, stride=p)   # patchify + project
self.proj2 = nn.Conv2d(128,      embed_dim,   kernel_size=1)              # expand to D
```

This adds a learned intermediate representation and changes the inductive bias of the first
layer. JiT-specific, not in PixelDiT.

---

## 5. Second EMA Tracker [main_jit.py:50–52, denoiser.py:32–34]

**Not in any existing MD file.**

The paper tracks a single EMA at decay 0.9999. The code maintains two:

```python
parser.add_argument('--ema_decay1', default=0.9999)   # used for evaluation
parser.add_argument('--ema_decay2', default=0.9996)   # faster decay, diagnostic
```

`ema_params2` is never used for sampling or evaluation — only `ema_params1` is loaded in both
`save_samples` and `evaluate`. The second EMA is dead weight in checkpoints and slightly
increases memory / EMA update cost. JiT-specific; not a correctness issue but an undocumented
divergence.

---

## 6. FinalLayer Defined But Never Used [model_jit.py:154–172]

**Not in any existing MD file.**

The `FinalLayer` class (the standard DiT output projection) is defined in full but is never
instantiated in `JiT.__init__` and never called in `JiT.forward`. The final output is produced
entirely by the pixel pathway (`pixel_norm_out → pixel_out`). Dead code — no functional impact,
but it is a maintenance hazard.

---

## Summary

| # | Difference | In diff_analysis.md? | In codex? | Severity |
|---|---|---|---|---|
| 1 | LR step-down missing (epoch 160) | Partial (LR magnitude only) | Yes | IMPORTANT |
| 2 | Depth mismatch (JiT-L=24 vs N=22, pixel blocks=3 vs 2/4) | No | Yes | Moderate |
| 3 | SwiGLU in patch blocks vs GELU | No | No | Minor–Moderate |
| 4 | Bottleneck patch embed (2-conv) | No | Yes | Minor/JiT-specific |
| 5 | Second EMA tracker (unused) | No | No | Cosmetic |
| 6 | `FinalLayer` dead code | No | No | Cosmetic |
