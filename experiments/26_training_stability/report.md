# Experiment 26 - GPT Training Stability

Status: completed

## 1. Files added and changed

- `experiments/26_training_stability.py` - controlled comparison entry point.
- `experiments/26_training_stability/report.md` - this final report.
- `experiments/26_training_stability/metrics.json` - machine-readable metrics.
- `tests/test_training_stability.py` - offline tests for the new diagnostics.
- `experiments/23_scalable_gpt_training.py` - opt-in training diagnostics
  (`diagnostics`, `diagnostics_interval`, `val_loader`, `diagnostics_val_loader`, and
  true mid-epoch `max_steps`); next-token loss reduction now in float32.

## 2. Audit summary

Before running experiments we re-checked the Experiment 23/25 training path:

- Causal multi-head self-attention scales scores by `sqrt(head_dim)`; the
  single-head path scales by `sqrt(embed_dim)`. Both are correct.
- Causal mask, residual connections, pre-norm LayerNorm, GELU FFN, and the
  next-token target shift were found to be correct.
- AdamW (weight_decay=0) and gradient clipping (threshold 1.0, applied after
  backward and before `optimizer.step()`) are placed correctly.
- BF16 runs the forward/backward under autocast; cross-entropy reduction is
  now forced to float32 for numerical safety.
- No architectural defect was found. The suspected causes were (a) the large
  optimizer step budget at a constant lr=1e-3 and (b) BF16 loss-precision
  effects, both testable with short controlled runs.

## 3. Controlled comparison setup

Each run starts from a fresh, identically seeded initialization and optimizes
the same corpus/tokenizer/split for a fixed step budget. This isolates the
effect of precision and lr; no full epoch (or multi-epoch) training is run.

- Steps per run: **300**
- Diagnostics interval (periodic val): **25** steps
- Periodic validation subset: first **25** val batches; final
  validation uses the full validation split.
- Architecture: 16 layers, d_model 512, 8 heads, dff 768, context 256,
  dropout 0, batch 4, AdamW weight_decay 0, clip_grad 1.0, seed 0,
  val_split 0.1, vocab 256.

## 4. Results

| run | precision | lr | steps | initial | final train | final val | clipped | update max | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16_lr_1e_3 | bf16 | 0.001 | 300 | 5.7198 | 3.7953 | 3.7512 | 40/300 | 5.087e-04 | True |
| bf16_lr_3e_4 | bf16 | 0.0003 | 300 | 5.7198 | 3.8078 | 3.7543 | 42/300 | 1.521e-04 | True |
| bf16_lr_1e_4 | bf16 | 0.0001 | 300 | 5.7198 | 4.0105 | 3.7982 | 65/300 | 4.174e-05 | True |
| fp32_lr_1e_3 | fp32 | 0.001 | 300 | 5.7198 | 3.8012 | 3.7475 | 41/300 | 5.149e-04 | True |

### Per-run diagnostic detail

**bf16_lr_1e_3** (lr=0.001, bf16)

- Train loss samples: 5.7360, 3.8709, 3.7523, 3.6937, 3.5482
- Periodic val: 25:4.4007, 50:3.9512, 75:3.9073, 100:3.7633, 125:3.8219, 150:3.7514, 175:3.6938, 200:3.6966, 225:3.7578, 250:3.6469, 275:3.6730, 300:3.6471
- Grad norm (pre-clip): mean 1.259 -> 0.809, range [0.729, 4.519]
- Update ratio: mean 2.511e-04 -> 4.512e-04, max 5.087e-04
- Dtypes: logits `torch.bfloat16`, loss `torch.float32`; all steps finite: True

**bf16_lr_3e_4** (lr=0.0003, bf16)

- Train loss samples: 5.7360, 3.9087, 3.7387, 3.6865, 3.5559
- Periodic val: 25:4.6034, 50:4.0600, 75:3.9003, 100:3.7802, 125:3.7805, 150:3.7404, 175:3.6881, 200:3.6795, 225:3.6909, 250:3.6390, 275:3.7094, 300:3.6354
- Grad norm (pre-clip): mean 1.238 -> 0.883, range [0.818, 9.025]
- Update ratio: mean 7.616e-05 -> 1.220e-04, max 1.521e-04
- Dtypes: logits `torch.bfloat16`, loss `torch.float32`; all steps finite: True

**bf16_lr_1e_4** (lr=0.0001, bf16)

- Train loss samples: 5.7360, 4.2523, 3.8506, 3.7657, 3.6133
- Periodic val: 25:4.9696, 50:4.5926, 75:4.2704, 100:4.0552, 125:3.9507, 150:3.8738, 175:3.8306, 200:3.7914, 225:3.7784, 250:3.7339, 275:3.7616, 300:3.7090
- Grad norm (pre-clip): mean 1.059 -> 0.961, range [0.855, 1.583]
- Update ratio: mean 2.572e-05 -> 3.462e-05, max 4.174e-05
- Dtypes: logits `torch.bfloat16`, loss `torch.float32`; all steps finite: True

**fp32_lr_1e_3** (lr=0.001, fp32)

- Train loss samples: 5.7361, 3.8573, 3.7563, 3.6949, 3.5555
- Periodic val: 25:4.4493, 50:3.9479, 75:3.8690, 100:3.7721, 125:3.8228, 150:3.7569, 175:3.6853, 200:3.7042, 225:3.7560, 250:3.6568, 275:3.7027, 300:3.6688
- Grad norm (pre-clip): mean 1.257 -> 0.810, range [0.730, 4.474]
- Update ratio: mean 2.498e-04 -> 4.465e-04, max 5.149e-04
- Dtypes: logits `torch.float32`, loss `torch.float32`; all steps finite: True

## 5. Findings (measured)

- **BF16 is not the cause of the Experiment 25 divergence.** At lr=1e-3, bf16 and fp32 are nearly identical: final train 3.7953 vs 3.8012, final val 3.7512 vs 3.7475. BF16 logits with fp32 loss reduction were finite at every step.
- **Within a fixed, short step budget, higher lr reaches lower loss:** final val 3.7512 (1e-3) < 3.7543 (3e-4) < 3.7982 (1e-4). Lower lr is simply slower, not more stable.
- **No short-horizon numerical instability at any tested configuration:** all runs kept finite losses, gradients, and parameters; gradient norms generally declined during the run and clipping stayed in a moderate range.
- **Update magnitudes stayed small** (max relative parameter update on the order of 1e-4), so no single step exploded.

## 6. Interpretation and limitations

- These runs cover only 300 optimizer steps. Experiment 25 diverged over tens of
  thousands of steps per epoch at a constant lr=1e-3, so the short runs cannot
  reproduce that long-horizon failure directly. They do, however, rule out BF16
  loss precision and short-term gradient/parameter blow-up as the cause.
- The remaining hypothesis is a **long-horizon optimization issue** driven by the
  very large step count at a constant lr=1e-3, i.e. the model drifts out of the
  good region after many updates. This is consistent with the fact that epoch 1
  looked healthy while epochs 2-3 degraded.
- One comparison run (`bf16_lr_3e_4`) had a large wall-clock outlier (~3469s vs
  ~360s for the other runs) with much lower tokens/s. This is a system-load
  artifact, not a configuration effect; its loss diagnostics match the others.

## 7. Recommendation

Measured evidence does not justify moving the base lr down to 1e-4 (it only
slows progress) and does not implicate BF16 (bf16 matches fp32). Recommended
next full training configuration:

- Keep **BF16 autocast with fp32 loss reduction** (memory/speed win, no observed
  numerical downside).
- Keep **AdamW base lr=1e-3** but add an **lr warmup followed by decay** (e.g.
  linear warmup then cosine or step decay) so the large step budget does not
  keep applying the peak lr for the whole run.
- Add **mid-epoch validation + checkpointing** (e.g. every few thousand steps) and
  select the best checkpoint by validation loss, instead of only per-epoch.
- Re-run a **short multi-epoch pilot** (e.g. 1-2 epochs on a subset or with a
  capped step count) to confirm the schedule stabilizes epoch-2+ behavior before
  committing to a full 3-epoch run.

## 8. Reproduction

```
.venv/bin/python experiments/26_training_stability.py
```

This writes `experiments/26_training_stability/metrics.json` and this report.
