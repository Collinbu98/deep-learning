# Experiment 23 - Scalable GPT Training Pipeline

Status: completed

## 1. Files added and changed

Added:

- `experiments/23_scalable_gpt_training.py` - configurable decoder-only GPT-style
  language model, next-token dataset, training loop, checkpoint save/restore,
  generation, CLI, and smoke-run driver.
- `experiments/23_scalable_gpt_training/report.md` - this final report.
- `tests/test_scalable_gpt_training.py` - offline validation tests.
- `checkpoints/23_scalable_gpt_training_best.pt` - smoke-run checkpoint.

Changed: none. Existing components were reused unchanged (no behavior change):
- `experiments/05_self_attention.py` (causal attention primitives).
- `src/minigrad/bpe_tokenizer.py` (deterministic BPE tokenizer).
- `src/minigrad/lm_eval.py` (held-out NLL helpers).

## 2. Architecture and parameter-count formula

The GPT-style model is a decoder-only Transformer with pre-normalization
residual blocks, causal self-attention, and a position-wise feed-forward
network with configurable width.

Per block parameters (Linear weights + biases included):

- attention: single head `3 * (d^2 + d)`; multi-head `4 * (d^2 + d)`.
- feed-forward: `(dff * d + dff) + (d * dff + d) = 2 * dff * d + dff + d`.
- two LayerNorms: `4 * d`.
- block total = attention + feed-forward + `4 * d`.

Whole model:

`total = vocab_size * d + context_len * d + n_layers * block + 2 * d + vocab_size * d + vocab_size`

where the final `vocab_size * d + vocab_size` covers the LM head and final
LayerNorm. The closed-form was verified against the live module parameter count
for two configurations (multi-head and single-head).

Example scale estimate: a GPT-2-small-like configuration
(`vocab_size=1000, n_layers=6, d_model=768, n_heads=12, dff=3072, context_len=768`)
yields approximately 30M parameters, in the range of the next milestone.

## 3. Dataset and split strategy

- Corpus: built-in smoke text for the smoke run; `--data <path>` for a real run
  (default local corpus `data/pg1342.txt`).
- Split: contiguous region in raw-text order before BPE fitting:
  `train = text[:int(len(text) * (1 - val_split))]`, `val = text[rest:]`.
- Tokenizer: `BPETokenizer(vocab_size).fit(train_text)`, then encoded train and
  validation streams; out-of-alphabet validation characters become UNK.
- Dataset: sliding next-token windows; `n_windows = max(0, len(ids) - context_len)`;
  window `s`: `x = ids[s:s+T]`, `y = ids[s+1:s+1+T]`.
- Loss: `CrossEntropyLoss` over full fixed-length windows (no padding needed);
  perplexity = `exp(mean token NLL)`.

## 4. Training configuration and smoke metrics

Smoke run (`--smoke`):

- Config: vocab_size=64 (BPE fit -> actual 45), n_layers=1, d_model=16,
  n_heads=1, dff=32, context_len=16, dropout=0.0, batch_size=2, epochs=2,
  lr=3e-3, val_split=0.2, clip_grad=1.0.
- Corpus: built-in smoke text, 122 chars; train 97, val 25.
- Tokens: train 70, val 21.
- Parameters: 3,725 (closed-form matched: `match=True`).
- Estimated parameter memory: 0.0149 MB; optimizer-state estimate: 0.0447 MB.
- Steps: 54 (2 epochs x 27 steps).
- Epoch metrics: train loss 3.5896 -> 3.0675; val loss 3.9827 (perplexity
  53.66) -> 3.9987 (perplexity 54.53); best val loss 3.98 at epoch 1.

Training loss decreased over the smoke run; validation loss stayed high
(perplexity ~54) as expected for a tiny random-init model on a 122-char corpus.

## 5. Checkpoint and generation verification

- Checkpoint written to `checkpoints/23_scalable_gpt_training_best.pt` with model
  state dict, model config, tokenizer dict, training state (step, optimizer state,
  RNG states), and args.
- Checkpoint restore validates config compatibility and raises on mismatch.
- Generation from the saved checkpoint was run without retraining and produced
  token output from the loaded model.

## 6. GPU/device validation and measured memory/performance

- Device resolution: auto -> `cuda:0`; explicit `cuda:99` -> RuntimeError (no
  silent CPU fallback); `cpu` supported.
- Actual device: `cuda:0` (AMD Radeon Graphics, ROCm; `torch.cuda.is_available()`
  True, device count 2, current device 0). Smoke run executed with device
  `cuda:0`.
- Memory: smoke model used ~0 MB allocated/reserved at capture time; estimated
  parameter memory 0.0149 MB for 3,725 params.
- Scale note: for a ~30M parameter model, parameter memory alone is ~120 MB
  (fp32); AdamW optimizer state ~2x parameters ~360 MB plus gradients ~120 MB;
  activations and batch buffers add further memory. Parameter memory alone is not
  the full training memory requirement.

## 7. Test count and full-suite result

- New tests: `tests/test_scalable_gpt_training.py` (10 tests).
- Full-suite result: **413 passed** (previous suite: 393 passed; +10 new).

## 8. Exact commands

```bash
# Smoke training run (tiny built-in corpus, tiny model)
.venv/bin/python experiments/23_scalable_gpt_training.py --smoke

# Real training run (uses local default corpus unless --data is given)
.venv/bin/python experiments/23_scalable_gpt_training.py --data data/pg1342.txt --epochs 3 --checkpoint checkpoints/23_scalable_gpt_training_best.pt

# Resume training from a checkpoint
.venv/bin/python experiments/23_scalable_gpt_training.py --resume checkpoints/23_scalable_gpt_training_best.pt --checkpoint checkpoints/23_scalable_gpt_training_resume.pt --epochs 3

# Generate from a checkpoint without retraining
.venv/bin/python experiments/23_scalable_gpt_training.py --checkpoint checkpoints/23_scalable_gpt_training_best.pt --generate-prompt "To "
```

## 9. Remaining limitations before the 30M-100M parameter training run

- Smoke task is tiny (3,725 params, 122-char corpus); no meaningful language
  competence is implied.
- Memory estimates are parameter-only estimates; activation/batch memory for
  larger runs must be measured.
- Real-corpus training loop not yet exercised at scale; feasibility of batch
  size, context length, and sequence packing for ~30M-100M parameter models
  remains to be established.
