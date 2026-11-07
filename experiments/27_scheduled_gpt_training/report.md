# Experiment 27 - Scheduled GPT Training

Status: completed

## 1. Files added and changed

- `experiments/27_scheduled_gpt_training.py` - scheduled training entry point.
- `experiments/27_scheduled_gpt_training/report.md` - this report.
- `experiments/27_scheduled_gpt_training/metrics.json` - machine-readable metrics.
- `experiments/27_scheduled_gpt_training/samples.json` - generated samples.
- `checkpoints/27_scheduled_gpt/` - latest/best/epoch checkpoints.
- `experiments/23_scalable_gpt_training.py` - added opt-in
  `WarmupCosineScheduler` (linear warmup + cosine decay to a nonzero minimum),
  scheduler stepping in `train_epoch`/`train_model`, scheduler + RNG state
  restoration, and `--lr-schedule` / `--warmup-steps` / `--min-lr` CLI flags.
- `tests/test_scheduled_gpt_training.py` - offline scheduler/restore tests.

## 2. Exact schedule and why it was selected

- Schedule: **linear-warmup + cosine-decay** over the planned optimizer-update budget.
- Peak lr: **1.0e-03** (kept equal to Experiment 25).
- Linear warmup: **2,000** optimizer steps
  (0.771% of total steps;
  2.31% of one epoch).
- Cosine decay from peak to a nonzero minimum lr of **1.0e-04**
  (10% of peak).
- Planned total optimizer steps: **259,257**
  (86,419 steps/epoch x 3 epochs).
- lr at step 0: 5.000e-07; at warmup end: 1.000e-03;
  at the final step: 1.000e-04.
- The scheduler advances once per optimizer update (after `optimizer.step()`),
  never on validation passes.

Why these choices (from Experiment 26 measurements):

- Experiment 26 showed BF16 and FP32 were nearly identical at lr=1e-3
  (final val 3.7512 vs 3.7475), so precision was not the divergence cause and
  the peak lr was kept at 1e-3.
- In the first 300 steps at lr=1e-3, pre-clip gradient norms spiked to ~4.5
  against a ~1.26 mean and ~13% of steps were clipped. A short warmup lets the
  early transient settle before reaching the peak lr; 2,000 steps is ~0.8% of
  the planned budget and ~2.3% of an epoch.
- Experiment 26's lr=1e-4 run stayed finite and still improved (val 3.798),
  supporting a nonzero 1e-4 floor instead of decaying to zero.

## 3. Loss and perplexity trajectories

### Per-epoch (full validation split)

| epoch | train loss | train ppl | val loss | val ppl | train steps | epoch seconds | lr at end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2756 | 1.3173 | 7.5525 | 1905.5301 | 86419 | 3045.0 | 7.813e-04 |
| 2 | 0.0547 | 1.0563 | 9.7011 | 16335.5287 | 86419 | 3034.5 | 3.282e-04 |
| 3 | 0.0407 | 1.0415 | 11.5906 | 108082.2544 | 86419 | 3059.5 | 1.000e-04 |

### Periodic intra-epoch measurements

Periodic validation uses a fixed subset of the first
100 validation batches (same methodology every
point); epoch-end rows above use the full validation split.

| step | epoch | step/epoch | lr | train loss | grad mean | grad max | clipped | update ratio | subset val | finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10802 | 1 | 10802 | 9.974e-04 | 1.4660 | 0.893 | 1.841 | 0.196 | 4.333e-01 | 4.3605 | yes |
| 21604 | 1 | 21604 | 9.872e-04 | 0.1879 | 0.562 | 1.170 | 0.000 | 4.825e-01 | 5.5249 | yes |
| 32406 | 1 | 32406 | 9.693e-04 | 0.1283 | 0.432 | 1.029 | 0.000 | 3.674e-01 | 6.2915 | yes |
| 43208 | 1 | 43208 | 9.442e-04 | 0.1045 | 0.373 | 0.761 | 0.000 | 2.610e-01 | 6.2311 | yes |
| 54010 | 1 | 54010 | 9.122e-04 | 0.0910 | 0.335 | 1.064 | 0.000 | 1.719e-01 | 6.1593 | yes |
| 64812 | 1 | 64812 | 8.740e-04 | 0.0820 | 0.304 | 1.084 | 0.000 | 1.396e-01 | 6.8095 | yes |
| 75614 | 1 | 75614 | 8.301e-04 | 0.0752 | 0.276 | 0.954 | 0.000 | 1.242e-01 | 6.8096 | yes |
| 86416 | 1 | 86416 | 7.813e-04 | 0.0697 | 0.253 | 0.858 | 0.000 | 1.014e-01 | 6.8340 | yes |
| 86419 | 1 | 86419 | 7.813e-04 | 0.0630 | 0.239 | 0.276 | 0.000 | 4.220e-04 | 6.8736 | yes |
| 97221 | 2 | 10802 | 7.285e-04 | 0.0652 | 0.232 | 0.829 | 0.000 | 7.964e-02 | 6.5406 | yes |
| 108023 | 2 | 21604 | 6.727e-04 | 0.0613 | 0.214 | 2.088 | 0.000 | 6.519e-02 | 7.1571 | yes |
| 118825 | 2 | 32406 | 6.146e-04 | 0.0582 | 0.196 | 0.767 | 0.000 | 5.408e-02 | 7.8067 | yes |
| 129627 | 2 | 43208 | 5.555e-04 | 0.0552 | 0.182 | 1.117 | 0.000 | 4.932e-02 | 8.1954 | yes |
| 140429 | 2 | 54010 | 4.963e-04 | 0.0527 | 0.167 | 0.721 | 0.000 | 3.868e-02 | 7.8894 | yes |
| 151231 | 2 | 64812 | 4.380e-04 | 0.0504 | 0.154 | 0.606 | 0.000 | 4.094e-02 | 8.6549 | yes |
| 162033 | 2 | 75614 | 3.816e-04 | 0.0484 | 0.141 | 0.929 | 0.000 | 3.486e-02 | 8.8847 | yes |
| 172835 | 2 | 86416 | 3.282e-04 | 0.0465 | 0.129 | 0.809 | 0.000 | 2.977e-02 | 8.6391 | yes |
| 172838 | 2 | 86419 | 3.282e-04 | 0.0477 | 0.163 | 0.222 | 0.000 | 1.459e-04 | 8.7090 | yes |
| 183640 | 3 | 10802 | 2.786e-04 | 0.0449 | 0.119 | 0.885 | 0.000 | 2.561e-02 | 9.1771 | yes |
| 194442 | 3 | 21604 | 2.338e-04 | 0.0433 | 0.110 | 0.437 | 0.000 | 2.095e-02 | 9.0962 | yes |
| 205244 | 3 | 32406 | 1.944e-04 | 0.0420 | 0.103 | 1.009 | 0.000 | 1.792e-02 | 9.7569 | yes |
| 216046 | 3 | 43208 | 1.612e-04 | 0.0408 | 0.097 | 1.756 | 0.000 | 1.380e-02 | 9.0660 | yes |
| 226848 | 3 | 54010 | 1.348e-04 | 0.0398 | 0.093 | 0.425 | 0.000 | 1.155e-02 | 10.2775 | yes |
| 237650 | 3 | 64812 | 1.156e-04 | 0.0389 | 0.090 | 0.643 | 0.000 | 9.358e-03 | 9.8036 | yes |
| 248452 | 3 | 75614 | 1.039e-04 | 0.0382 | 0.088 | 1.745 | 0.000 | 5.972e-03 | 10.3013 | yes |
| 259254 | 3 | 86416 | 1.000e-04 | 0.0376 | 0.087 | 0.515 | 0.000 | 7.771e-03 | 10.1869 | yes |
| 259257 | 3 | 86419 | 1.000e-04 | 0.0353 | 0.081 | 0.102 | 0.000 | 4.420e-05 | 10.1951 | yes |

## 4. Did training and validation improve consistently?

- Train loss improved every epoch: **True** (0.2756 -> 0.0547 -> 0.0407).
- Validation loss improved every epoch: **False** (7.5525 -> 9.7011 -> 11.5906).
- All periodic finite-value checks passed: **True**.
- Best validation epoch: **1** (val loss 7.5525).
- Best periodic subset validation point: step 10,802 (subset val loss 4.3605; the subset is not directly comparable to the full-val epoch rows).

## 5. Was the Experiment 25 failure avoided?

| epoch | Exp25 train | Exp27 train | Exp25 val | Exp27 val |
| --- | --- | --- | --- | --- |
| 1 | 2.8470 | 0.2756 | 3.8780 | 7.5525 |
| 2 | 4.5317 | 0.0547 | 4.6669 | 9.7011 |
| 3 | 4.8949 | 0.0407 | 5.0388 | 11.5906 |

Measured: Experiment 25's train and validation losses both rose after
epoch 1. Experiment 27 does not reproduce a rising training loss (it falls
every epoch), but its full-validation loss rises monotonically after epoch 1
and ends far above Experiment 25's. The epoch-1 full-val result is already
well above the early subset-val minimum (section 11), so this is severe
overfitting rather than the Experiment 25 optimization breakdown.

Baseline caveat: Experiment 25 predates the FP32 cross-entropy reduction
added while running Experiment 26 (git history commits both together).
Its 3.878/4.667/5.039 trajectory is therefore not an apples-to-apples
baseline for the current code path; Experiment 26's current-code 300-step
runs (subset val 3.647-3.751) are the relevant short-horizon reference.

## 6. Generated text across training

Generation settings: {"max_new_tokens": 40, "temperature": 0.8, "top_k": null}.

### Stage initial

- Prompt 'It is a truth universally true that' -> It is a truth universally true thather bethbut Eliza, and ed 9de‘ou.”7d be 3lizakDXat be Elizabethll dalceosit sed y been1x2liz; and 
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudiceher bethbut en , and ch 9P‘ou.”7d x43lizbethld Xat be Elizabethll dirceosit sed y urenXx2liz; and 
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyDbethbut Eliza, and edwith y‘ou.”twaha3lizbethld Xat be Elizabethll dby ceout sed y urshe Xx2liz; and 
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost andher bethbut Eliza, and ed 9de‘ou.”7d beha3lizakDXhabe Elizabethll dalceosit sed y been1x2liz; and 
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughher bethbut Eliza, and ed9P‘ai.”twvery  3lizbethld Xto be Elizabethll dirceosit sed y Ren1but 2liz; and 

### Stage epoch_1

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
it would be wiser to give it to make her fresh as soon as he formerly in
anything, and m
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice

Author: Jane Austen


        
Relegance, and during the two part of a prett
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy. If I
had no people directly rejoice in me last nightcapable of d
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
unallowing. You must come and make me to speak place at this
time of ye
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plught.

She could not but two letters gan instant, she felt certainly on that he
had of being 

### Stage epoch_2

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
every private care; and covering her face with her handkerchief,
Elizabeth was soon lost 
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice
    
This eBook is for the use of anyone anywhere in the United States and
mos
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy. Will you be very angry with me, my dear
     Lizzy, if I take this opportunit
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
She thought it bornet in Lady Catherine contracted with the sizement
and the gentlem
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughters, she flattered herself that her wishes did
not err. But when she read and re-read, 

### Stage epoch_3

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
every moment more unconcerned and indifferent. Your profusion makes me
sa
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice
    
This eBook is for the use of anyone anywhere in the United States and
mos
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy. Will you be spring! I will not take it on myself.”

Then turned round, he 
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
Miss Bingley. I think I have heard you say that you know them.”

“I know them a litt
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughter must be done by him! England
with an air of followed her to speak when

### Stage best

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
it would be wiser to give it to the society of the
neighbourhood, nor incl
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice

Here to Meryton, the
neighbourhood, the society, appearing highly ple
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy,” said Darcy, “I have been mistaken
daughter. Good graciously.”

S
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
unallantry _must_ feelings pride, is such a cause humble practi
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plught.

She could not, however, in her partiality of such a conversation, he made hi

Interpretation: compare recognizable words/structure, repetition, and
degeneration across stages, and check whether quality tracks validation loss.

## 7. Throughput, memory, and duration

- Total optimizer steps: 259,257
- Tokens processed: 265,477,632
- Measured training seconds (epoch walls incl. validation): 9139.0
- Total run wall seconds: 9171.1
- Throughput: 29048.88 tokens/s
- Peak GPU memory: 1188.53 MB allocated / 1218.45 MB reserved
- Epoch 1: 3045.0s, 88,492,544 tokens
- Epoch 2: 3034.5s, 88,492,544 tokens
- Epoch 3: 3059.5s, 88,492,544 tokens

## 8. Checkpoints and resume

- Latest: `/home/collin/deep-learning/checkpoints/27_scheduled_gpt/latest.pt`
- Best: `/home/collin/deep-learning/checkpoints/27_scheduled_gpt/best.pt` (epoch 1)
- Epoch checkpoints: {"1": "/home/collin/deep-learning/checkpoints/27_scheduled_gpt/epoch_1.pt", "2": "/home/collin/deep-learning/checkpoints/27_scheduled_gpt/epoch_2.pt", "3": "/home/collin/deep-learning/checkpoints/27_scheduled_gpt/epoch_3.pt"}
- Checkpoints include model weights, optimizer state, scheduler state, RNG
  state (Python/CPU/CUDA), step counters, and configuration metadata.

```bash
.venv/bin/python experiments/27_scheduled_gpt_training.py
.venv/bin/python experiments/27_scheduled_gpt_training.py --resume checkpoints/27_scheduled_gpt/epoch_2.pt
```

## 9. Readiness and recommendation

Read section 4/5 for the measured trend before reusing this base model.
If all three epochs improved monotonically and stayed finite, the scheduled
recipe resolves the Experiment 25 deterioration at this budget and the best
checkpoint is suitable as the base for the next stage. If any epoch worsened,
the periodic table identifies where; the next diagnostic should then test a
longer warmup, a lower peak lr, or a shorter effective step budget rather than
scaling up. All statements above are measured unless explicitly marked as
interpretation.

Measured result of this run: training loss fell every epoch, but full-val
loss rose every epoch and the subset-val trajectory rose within epoch 1.
Interpretation: the schedule removed the Experiment 25 train-loss blow-up but
did not produce a good final model; the model overfits before the first
periodic validation point. The next run should validate far more frequently
early (e.g. every 250-1,000 steps), keep the best checkpoint by full
validation, and consider a shorter budget or stronger regularization rather
than three full epochs.

## 10. Full test-suite result

438 passed (full suite via `.venv/bin/python -m pytest -q`; latest run 37.55s). This includes 10 new scheduler/restore/experiment-27 tests in `tests/test_scheduled_gpt_training.py`.

## 11. Supplementary early-validation pilot

The main run validated only eight times per epoch (every ~10,802
steps), so it could not resolve the early optimum. This supplementary
pilot replays the same seed/configuration for a short budget with
validation every 100 steps, using the first 25 validation batches.

- Pilot steps: 3,000; validation interval: 100; subset val batches: 25.
- Full-validation loss at step 3,000: 2.6503.
- Best subset-val point: step 3,000, val 2.5171.
- Schedule caveat: the pilot's planned total is only 3,000 steps, so after warmup its lr decays faster
  than the main run's; steps 0-2,000 use the same warmup and are the
  directly comparable portion.

| step | lr | train loss | subset val |
| --- | --- | --- | --- |
| 100 | 5.000e-05 | 5.1951 | 4.9312 |
| 200 | 1.000e-04 | 4.3427 | 4.0340 |
| 300 | 1.500e-04 | 3.7917 | 3.7569 |
| 400 | 2.000e-04 | 3.6455 | 3.7271 |
| 500 | 2.500e-04 | 3.5927 | 3.6610 |
| 600 | 3.000e-04 | 3.5411 | 3.5942 |
| 700 | 3.500e-04 | 3.5009 | 3.6355 |
| 800 | 4.000e-04 | 3.4680 | 3.5470 |
| 900 | 4.500e-04 | 3.4605 | 3.5545 |
| 1000 | 5.000e-04 | 3.3854 | 3.4495 |
| 1100 | 5.500e-04 | 3.1789 | 3.1983 |
| 1200 | 6.000e-04 | 2.9888 | 3.1092 |
| 1300 | 6.500e-04 | 2.8436 | 3.0447 |
| 1400 | 7.000e-04 | 2.7384 | 2.8983 |
| 1500 | 7.500e-04 | 2.6397 | 2.9104 |
| 1600 | 8.000e-04 | 2.6223 | 2.7941 |
| 1700 | 8.500e-04 | 2.5644 | 2.9071 |
| 1800 | 9.000e-04 | 2.5411 | 2.8558 |
| 1900 | 9.500e-04 | 2.4735 | 2.8765 |
| 2000 | 1.000e-03 | 2.4744 | 2.7597 |
| 2100 | 9.784e-04 | 2.4455 | 2.7313 |
| 2200 | 9.147e-04 | 2.4029 | 2.6664 |
| 2300 | 8.153e-04 | 2.3325 | 2.7154 |
| 2400 | 6.899e-04 | 2.2602 | 2.6544 |
| 2500 | 5.507e-04 | 2.1901 | 2.6134 |
| 2600 | 4.115e-04 | 2.0999 | 2.5994 |
| 2700 | 2.858e-04 | 2.0154 | 2.5361 |
| 2800 | 1.861e-04 | 1.9619 | 2.5218 |
| 2900 | 1.221e-04 | 1.8709 | 2.5239 |
| 3000 | 1.000e-04 | 1.8357 | 2.5171 |

Measured: subset val reached its minimum in this pilot at step
3,000 and was already far below the main
run's first periodic subset-val point (step 10,802, val 4.3605).
Interpretation: the useful checkpoint lies in the first few thousand
steps, before the main run's validation cadence sampled the model.
