# Experiment 24 - GPT Training Performance and Scale Feasibility

Status: completed

## 1. Files added and changed

Added:

- `experiments/24_training_benchmarks.py` - short-run GPU benchmark driver
  (reuses Experiment 23 pipeline).
- `experiments/24_training_benchmarks/report.md` - this final report.
- `experiments/24_training_benchmarks/benchmark_report.json` - default benchmark
  report artifact.
- `tests/test_training_benchmarks.py` - offline tests for benchmark utilities.

Changed: none.

## 2. Baseline and setup

- Device: `cuda:0` (AMD Radeon Graphics, compute capability 11.0; ROCm via
  PyTorch `torch.cuda`; torch 2.9.1+rocm6.4).
- GPU memory (device 0): free ≈ 16,953 MB, total ≈ 17,163 MB.
- Baseline benchmark (Experiment 23 smoke configuration) completed with finite
  losses and short step times; this establishes the harness baseline.

## 3. Measured benchmark results

All executed configurations completed successfully with finite losses and
working gradients/optimizer updates.

| Run | Params | Context | Batch | Steps | Median step (s) | Tokens/s | Peak alloc (MB) | Peak res (MB) | Final loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke baseline | 4,352 | 16 | 2 | 3 | 0.0022 | 14,793 | 146.9 | 148.9 | 3.37 |
| 30M fp32 | 29,840,640 | 256 | 4 | 5 | 0.0673 | 15,208 | 1048.9 | 1205.9 | 2.38 |
| 30M fp32 (ctx=512) | 29,971,712 | 512 | 4 | 3 | 0.1416 | 14,463 | 1879.1 | 2053.1 | 4.48 |
| 30M bf16 | 29,840,640 | 256 | 4 | 3 | 0.0320 | 31,953 | 1059.7 | 1086.3 | 3.52 |
| 30M fp32 (ctx=128, batch=8) | 29,775,104 | 128 | 8 | 3 | 0.0651 | 15,742 | 981.6 | 1119.9 | 4.02 |
| 50M fp32 | 50,861,312 | 256 | 2 | 5 | 0.0642 | 7,973 | 1173.6 | 1187.0 | 1.97 |
| 100M fp32 | 93,324,032 | 512 | 1 | 5 | 0.0861 | 5,947 | 2040.3 | 2216.7 | 1.06 |

Measured facts:

- All losses finite; gradients and AdamW updates completed without error.
- Measured throughput for the ~30M fp32 config: ~15,208 tokens/s
  (batch 4 x context 256 = 1024 tokens/step).
- Mixed precision (bf16 autocast) supported and finite; median step dropped to
  0.0320 s (~2.1x faster, ~31,953 tokens/s) with similar memory behavior.
- Peak GPU memory stayed well below the 16 GB VRAM budget in all runs; largest
  observed peak reserved ≈ 2.2 GB (100M config).

## 4. Failed configurations and reasons

No benchmark configuration failed during execution. All runs reported
`status: completed` with finite final losses.

## 5. Duration estimates.

Using measured tokens/second and the train-split token counts:

- ~30M fp32: 345,930 train tokens @ 15,208 tokens/s ≈ 22.8 s/epoch; bf16 ≈ 10.8 s/epoch.
- ~50M fp32: 345,930 train tokens @ 7,973 tokens/s ≈ 43.4 s/epoch.
- ~100M fp32: 271,580 train tokens @ 5,947 tokens/s ≈ 45.6 s/epoch.

Assumptions: one epoch = all train tokens; validation/checkpoint overhead not
included. Estimates are extrapolations from 3-step measurements.

## 6. Recommendation for the first real training run.

Recommended starting configuration:

- Model: ~30M parameter GPT-style model (actual 29,840,640 params):
  `vocab_size=256, n_layers=16, d_model=512, n_heads=8, dff=768, context_len=256, dropout=0.0`.
- Precision: bf16 autocast (measured ~2.1x faster, finite loss/grads).
- Batch size: 4; context length: 256; learning rate: 1e-3; AdamW; clip_grad 1.0.
- Token budget: ~3 epochs over the train split ≈ 1.04M tokens (≈ 350k tokens/epoch;
  ≈ 11 s/epoch with bf16), which is short and informative relative to smoke.
- Validation: every 100 steps; checkpoint: every 500 steps (or best-val).

Rationale: measured throughput ~32k tokens/s, peak memory ~1.06 GB, loss decreasing,
and capacity in the requested ~30M-50M range while keeping a large VRAM margin.

## 7. Reproduction commands.

```bash
.venv/bin/python experiments/24_training_benchmarks.py --smoke --steps 3
.venv/bin/python experiments/24_training_benchmarks.py --data data/pg1342.txt \
    --vocab_size 256 --n_layers 16 --d_model 512 --n_heads 8 --dff 768 \
    --context_len 256 --batch_size 4 --steps 5
.venv/bin/python experiments/24_training_benchmarks.py --data data/pg1342.txt \
    --vocab_size 256 --n_layers 24 --d_model 512 --n_heads 8 --dff 1024 \
    --context_len 256 --batch_size 2 --steps 5
.venv/bin/python experiments/24_training_benchmarks.py --data data/pg1342.txt \
    --vocab_size 512 --n_layers 13 --d_model 768 --n_heads 12 --dff 3072 \
    --context_len 512 --batch_size 1 --steps 5 --mixed-precision
```

## 8. Final test count and full-suite result.

- Full-suite result: **417 passed**.

## 9. Remaining limitations.

- Benchmarks used 3 timed steps; steady-state throughput not fully established.
- Duration estimates ignore validation/checkpoint overhead.
- No long training run performed (by design).
