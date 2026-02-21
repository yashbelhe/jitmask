# Additional PixelDiT vs Current Code Differences (Codex)

This file lists extra paper-vs-code differences identified beyond `diff_analysis.md`, **excluding EMA and inference settings**.

## 1) Model family / depth configuration mismatch

Paper ImageNet variants (Table 2 / Table 6):
- PixelDiT-B: patch depth `N=12`, pixel depth `M=2`, hidden size `D=768`
- PixelDiT-L: patch depth `N=22`, pixel depth `M=4`, hidden size `D=1024`
- PixelDiT-XL: patch depth `N=26`, pixel depth `M=4`, hidden size `D=1152`

Current code:
- Global default `num_pixel_blocks=3` (`main_jit.py:64`)
- `JiT_L_16` uses patch depth `24` (`model_jit.py:535`)
- No PixelDiT-XL-style model; largest is JiT-H (`D=1280`) (`model_jit.py:543`)

Paper refs:
- `references/pixeldit_2511.20645v1.txt:1316`
- `references/pixeldit_2511.20645v1.txt:1324`
- `references/pixeldit_2511.20645v1.txt:1331`

## 2) Learning-rate schedule mismatch (policy, not just base LR)

Paper training policy:
- Constant `1e-4` for first 160 epochs
- Then step down to `1e-5`

Current code:
- Warmup + schedule mode (`constant` or `cosine`)
- No explicit 160-epoch step-down policy

Code refs:
- `main_jit.py:35`
- `main_jit.py:45`
- `util/lr_sched.py:9`

Paper refs:
- `references/pixeldit_2511.20645v1.txt:1398`
- `references/pixeldit_2511.20645v1.txt:1405`

## 3) Batch-size protocol mismatch

Paper:
- Batch size `256` for ImageNet class-conditioned setup.

Current code:
- Default `batch_size=128` per GPU and scales effective batch with world size.
- This can differ substantially from paper unless run config is matched carefully.

Code refs:
- `main_jit.py:37`
- `main_jit.py:186`

Paper ref:
- `references/pixeldit_2511.20645v1.txt:1393`

## 4) Patch embedding operator mismatch

Paper formalism:
- Single patch projection `s0 = W_patch x_patch`.

Current code:
- Two-stage bottleneck patch embedding:
  - `Conv2d(in_chans -> pca_dim, kernel=p, stride=p)`
  - `Conv2d(pca_dim -> D, kernel=1)`

This is a meaningful architecture difference from the stated PixelDiT formulation.

Code refs:
- `model_jit.py:29`
- `model_jit.py:30`

Paper ref:
- `references/pixeldit_2511.20645v1.txt:430`
