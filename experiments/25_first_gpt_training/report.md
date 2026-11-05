# Experiment 25 - First Real GPT Training Run

Status: completed

## 1. Files added and changed

- `experiments/25_first_gpt_training.py` - first real GPT training entry point
- `experiments/25_first_gpt_training/report.md` - this final report.
- `experiments/25_first_gpt_training/metrics.json` - machine-readable metrics.
- `experiments/25_first_gpt_training/samples.json` - generated sample artifact.
- `tests/test_first_gpt_training.py` - offline tests for the new orchestration,
  configuration, artifact, and generation code.
- `experiments/23_scalable_gpt_training.py` - added optional bf16 mixed-precision
  training path (`use_mixed_precision` argument on `train_epoch` / `train_model`,
  `--mixed-precision` CLI flag); default behavior unchanged.

## 2. Setup

1. Exact architecture, parameter count, tokenizer, corpus, split, and seed.

- Architecture: decoder-only, causal, next-token-prediction GPT-style
  transformer (pre-normalization residual blocks, multi-head causal
  self-attention, GELU feed-forward, pre-norm, final LayerNorm, token +
  learned-absolute-position embeddings, LM head). Configuration: 16
  layers, d_model 512, 8 heads (head dim 64), feed-forward width 768,
  context length 256, dropout 0.0.
- Parameter count: 29,840,640 (closed-form estimate and live module count
  both matched: parameter_count_matches true).
- Tokenizer: existing character-level BPE tokenizer from
  src/minigrad/bpe_tokenizer.py (type bpe_char). Fit on the training
  portion of the corpus with target vocabulary size 256; actual
  vocabulary size 256 (alphabet 92 + 163 learned merges + reserved UNK
  id 255). Tokenizer reused from the Experiment 8 implementation; no new
  tokenizer was trained or downloaded.
- Corpus: local data/pg1342.txt (Project Gutenberg, Pride and Prejudice),
  748,167 characters. Contiguous train/validation split at val_split 0.1
  (train 673,350 chars / 345,930 tokens; validation 74,817 chars / 40,198
  tokens), taken before tokenization so BPE merges never saw held-out text.
- Seed: 0 (Python and torch CUDA RNG seeded before model init).

2. Precision, optimizer, learning rate, batch size, context length, and
   epoch count.

- Precision: bf16 mixed precision via torch.autocast(device_type cuda:0,
  dtype torch.bfloat16) wrapping forward + backward (the mode validated in
  Experiment 24; wired through the Experiment 23 pipeline for Experiment
  25). Losses and gradients remained finite throughout; no precision change
  or hyperparameter change was made during the run.
- Optimizer: AdamW (lr 1e-3, weight_decay 0.0), gradient clipping (clip_grad
  1.0), cross-entropy loss over fixed-length next-token windows.
- Batch size: 4. Context length: 256. Epochs: 3.

## 3. Training and validation metrics after every epoch.

| epoch | train loss | train perplexity | val loss | val perplexity | train tokens | train steps | epoch seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.8470 | 17.2356 | 3.8780 | 48.3290 | 88492544 | 86419 | 3034.5 |
| 2 | 4.5317 | 92.9129 | 4.6669 | 106.3720 | 88492544 | 86419 | 3043.2 |
| 3 | 4.8949 | 133.6026 | 5.0388 | 154.2912 | 88492544 | 86419 | 2947.1 |

- Best validation checkpoint: epoch 1 (val loss
  3.8780, perplexity 48.3290).
- Final validation loss: 5.0388 | perplexity
  154.2912.
- Total optimizer steps: 259257; tokens processed: 265,477,632.

## 4. Actual duration, throughput, and peak memory.

Measured during the run (not estimates):

- Epoch 1: 3034.5s (train loss 2.8470,
  val loss 3.8780, val perplexity 48.3290).
- Epoch 2: 3043.2s (train loss 4.5317,
  val loss 4.6669, val perplexity 106.3720).
- Epoch 3: 2947.1s (train loss 4.8949,
  val loss 5.0388, val perplexity 154.2912).
- Total measured training seconds: 9024.8.
- Total run wall seconds: 9387.4.
- Measured throughput: 29416.33 tokens/s over
  265,477,632 tokens.
- Peak GPU memory: 1059.73 MB
  allocated / 1086.32 MB reserved.

Note: the Experiment 24 report estimated roughly 11 seconds per epoch for
this configuration; the measured epoch duration was about 50 minutes. The
estimate used unique token counts, ignoring that each epoch scores all
sliding-window token positions (~88.5M per epoch).

## 5. Generated samples from multiple training stages.

Generation settings: {"max_new_tokens": 40, "temperature": 0.8, "top_k": null}.

- Stage initial:
  - Prompt 'It is a truth universally true that' -> It is a truth universally true thather bethbut Eliza, and tt·P‘ai.”7N(ha3lizbethwhiXto unElizabethll Pby ceoenis sed y ”reXbut 2lizvery 
  - Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudiceher bethbut Eliza, and edwith P‘ai.”tofhem3lizbethld Xto unElizabethll , and alut ositat o y beenXbut Wlizdi
  - Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyher bethbut Eliza, and le9A‘ai.”tofheri3liz awhiXto unElizabethll , and by ceoentat ed y Gen1but 1liz; and 
  - Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost andher bethbut Eliza, and tt·A‘ai.”7N(ha3lizbethwhiXto unElizabethll Pby ceositat o y ”reXbut 2lizvery 
  - Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughher bethbut Eliza, and le9A‘ai.”7wœ23liz aecXat nowElizabethll dhe ut ositat ed y pen1but 2lizvery 
- Stage epoch_1:
  - Prompt 'It is a truth universally true that' -> It is a truth universally true thate
but she had perrare.”--shas of ss and to be ill dispoon taned thebeended pwith hi
  - Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudiceley. I had fordon on to
awhall here of the deard he ss of his frul, in mes her hersing,

  - Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy
pear thing in the men st of Mish to ofshigh, revevevitever poin be don
  - Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and, nree
eannor of theone of the prolawentrie:lmind to
gincafurded ou
  - Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughty seen theable, be
the
enld mup the me
cor of aued of
of at it star an of ourh
- Stage epoch_2:
  - Prompt 'It is a truth universally true that' -> It is a truth universally true thatorrirpeppan Sof         Jic the to the cle thather pouall agh be d lised ar that the l
  - Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudicerevy of in renhanly le. d heav; the the Tuzucleent in; insould of Gene iny, to se
  - Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyahe not Iararre that to s. c, mo whifucromisreenaruf thonou
s, thas, wo
  - Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost andcadss delongall beorled thetstidek ti

toe ]on w hsich as, and bee wt or
  - Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughed a mdrenertaousewistionne it with I I beled a.”fe dae daws uve the in 
- Stage epoch_3:
  - Prompt 'It is a truth universally true that' -> It is a truth universally true that
mdeus , e and andpcomeon ismoumttanI usdIs dpi bweecarkut by p
  - Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and PrejudicealHagof arvth llarmheer isecin
penschonpwhad , to to his ohis  uwor
 geed 
  - Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy smssiy, gent kHBoforristing yt 
sehait ishuforts 

elmevn“is, rt i
  - Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost andll obe wpsf
ee e erhad eds 
eseirasm
eing tutheed arll ain sbfnot em
  - Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plugha le uud ationthr-reothe mt?hedeell
y 
ce hitatin en s orasthat sof an I t
- Stage best:
  - Prompt 'It is a truth universally true that' -> It is a truth universally true thatt of her
overe they of been my Cottinsure able ucertoneent.

Liss were they. 
  - Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice; and her compaidoloout
daughthok ie. I had not ind your go p; and his pard is 
  - Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy
youappear ing to s by been her, I  applecties, he thought serle my that it dare
wert 
  - Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost ande ise vaying rough
his under; th, sing al wis, or nertioued able by duremite
  - Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plugh
that his lookle.the bointo the uls cerve to uss barly
comproause in airsake 

Qualitative comparison: initial, per-epoch, and best-checkpoint samples were
compared with fixed generation settings; see samples.json.

## 6. Checkpoint paths and commands for training, resuming, and generation.

- Best: `/home/collin/deep-learning/checkpoints/25_first_gpt/best.pt`
- Latest: `/home/collin/deep-learning/checkpoints/25_first_gpt/latest.pt`
- Epoch checkpoints: {"1": "/home/collin/deep-learning/checkpoints/25_first_gpt/epoch_1.pt", "2": "/home/collin/deep-learning/checkpoints/25_first_gpt/epoch_2.pt", "3": "/home/collin/deep-learning/checkpoints/25_first_gpt/epoch_3.pt"}

```bash
.venv/bin/python experiments/25_first_gpt_training.py
.venv/bin/python experiments/25_first_gpt_training.py --resume checkpoints/25_first_gpt/best.pt --checkpoint-dir checkpoints/25_first_gpt
.venv/bin/python experiments/25_first_gpt_training.py --generate-prompt "It is a truth"
```

## 7. Full test-suite result.

- Full suite: **424 passed** (previous baseline: 417 passed; +7 new offline
  tests in `tests/test_first_gpt_training.py`).

## 8. Reasoned recommendation.

Do not proceed to a longer training run from this recipe as configured. The
first real run at the Experiment 24 recommended settings (lr=1e-3, 3 epochs)
showed: (a) validation worsened after epoch 1 and the best checkpoint is
epoch 1, (b) training loss/perplexity diverged upward in later epochs, and
(c) generated samples degraded. Next steps: restart training from random
initialization with a reduced learning rate (e.g. 3e-4 or 1e-4) and/or shorter
effective step budget, verify stability, then scale epochs.
