# Experiment 28 - Early Generalization and Training Horizon

Status: completed

## 1. Deliverables

- `experiments/28_early_generalization.py` - experiment entry point.
- `experiments/28_early_generalization/report.md` - this report.
- `experiments/28_early_generalization/metrics.json` - all trajectories and results.
- `experiments/28_early_generalization/samples.json` - generated text.
- `checkpoints/28_early_generalization/h{3000,6000,12000}/best.pt` and `final.pt`.
- `tests/test_early_generalization.py` - offline CPU tests.

## 2. Phase 1 audit: Experiment 27 pilot comparability

The Experiment 27 supplementary pilot ran 3,000 steps with 2,000 warmup steps (66.7% of its budget), validation every 100 steps on the first 25 val batches, and reached subset val 2.5171 / full val 2.6503.

Shared with Experiment 28:
- 30M architecture, tokenizer, vocabulary, and pg1342 split
- seed 0 initialization via _set_global_seed before model construction
- data ordering reset to seed 0 immediately before training
- BF16 autocast with FP32 cross-entropy reduction
- AdamW, peak lr 1e-3, weight_decay 0, clip_grad 1.0
- full-validation implementation (exp23.validate_epoch)

Differences:
- pilot warmup was 2,000 of 3,000 steps (66.7%); Experiment 27 used 0.77% and Experiment 28 scales that fraction
- pilot decayed from peak to 1e-4 over the final 1,000 steps; Experiment 28 horizons decay over their full planned budget
- pilot evaluated the first 25 val batches; Experiment 28 uses the first 100
- pilot generated samples with one prompt and 8 new tokens; Experiment 28 uses the five default prompts and 40 new tokens

Use: The pilot is used only as motivation and as a coarse lower bound on early validation loss, not as a direct horizon comparison.

## 3. Fixed setup and schedules

- Architecture: 16 layers, d_model 512, 8 heads, dff 768, context 256, dropout 0.0, batch 4, vocab 256.
- Data: `/home/collin/deep-learning/data/pg1342.txt`; train/val chars 673,350/74,817; tokens 345,930/40,198.
- Optimizer: AdamW, peak lr 1.0e-03, weight decay 0.0, clip_grad 1.0, BF16 autocast, seed 0.
- Warmup: same fraction as Experiment 27 (Experiment 27: 2,000 / 259,257 planned updates); warmup = max(1, round(warmup_fraction * horizon)).
- Nonzero LR floor: 10% of peak (1.0e-04).
- Validation every 250 steps on the first 100 val batches; full validation for each saved best and final checkpoint.

| horizon | warmup steps | warmup % | lr step 0 | lr warmup end | lr final |
| --- | --- | --- | --- | --- | --- |
| 3,000 | 23 | 0.77% | 4.348e-05 | 1.000e-03 | 1.000e-04 |
| 6,000 | 46 | 0.77% | 2.174e-05 | 1.000e-03 | 1.000e-04 |
| 12,000 | 93 | 0.78% | 1.075e-05 | 1.000e-03 | 1.000e-04 |

## 4. Results

| horizon | best step | best subset val | best full val | best full ppl | train subset @ best | gap | lr @ best | final full val | deteriorated | seconds | tok/s | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3,000 | 2,750 | 2.4682 | 2.7804 | 16.1256 | 2.4827 | 0.2977 | 1.156e-04 | 2.7666 | False | 139.4 | 22043.8 | 1185.9 |
| 6,000 | 4,750 | 2.4032 | 2.7956 | 16.3726 | 2.1682 | 0.6274 | 1.944e-04 | 2.8359 | True | 272.1 | 22577.7 | 1185.9 |
| 12,000 | 5,500 | 2.3813 | 2.7911 | 16.2991 | 2.0666 | 0.7245 | 6.147e-04 | 4.1301 | True | 539.0 | 22797.3 | 1186.4 |

Measured best among `best.pt` checkpoints: horizon **3,000**, step **2,750**, full-val loss **2.7804**, perplexity **16.1256**.
Measured best among all saved `best.pt`/`final.pt` checkpoints: horizon **3,000**, step **3,000**, `final.pt`, full-val loss **2.7666**, perplexity **15.9047**.

## 5. Complete validation trajectories

Training loss is the token-weighted mean over the preceding validation interval; validation loss uses the fixed first-batch subset.

### Horizon 3,000

| step | lr | train loss | val loss | val ppl | grad mean | grad max | clipped | elapsed s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 9.873e-04 | 3.8469 | 3.5200 | 33.7841 | 0.975 | 11.619 | 0.184 | 11.1 |
| 500 | 9.444e-04 | 3.5015 | 3.4196 | 30.5580 | 0.815 | 1.669 | 0.024 | 22.5 |
| 750 | 8.742e-04 | 3.2035 | 2.9455 | 19.0195 | 0.851 | 1.459 | 0.020 | 33.9 |
| 1,000 | 7.815e-04 | 2.8690 | 2.8380 | 17.0815 | 0.869 | 1.181 | 0.032 | 45.4 |
| 1,250 | 6.729e-04 | 2.6750 | 2.7174 | 15.1402 | 0.853 | 1.171 | 0.012 | 57.0 |
| 1,500 | 5.557e-04 | 2.5682 | 2.6566 | 14.2471 | 0.848 | 1.172 | 0.012 | 68.5 |
| 1,750 | 4.381e-04 | 2.4936 | 2.6154 | 13.6727 | 0.855 | 0.976 | 0.000 | 80.0 |
| 2,000 | 3.283e-04 | 2.3909 | 2.5482 | 12.7835 | 0.866 | 1.042 | 0.012 | 91.6 |
| 2,250 | 2.338e-04 | 2.3053 | 2.5149 | 12.3655 | 0.892 | 1.062 | 0.008 | 103.2 |
| 2,500 | 1.612e-04 | 2.2249 | 2.4807 | 11.9491 | 0.906 | 1.044 | 0.012 | 114.8 |
| 2,750 * | 1.156e-04 | 2.1557 | 2.4682 | 11.8014 | 0.930 | 1.047 | 0.024 | 126.3 |
| 3,000 | 1.000e-04 | 2.0887 | 2.4703 | 11.8255 | 0.955 | 1.195 | 0.124 | 139.4 |

`*` marks the best subset-validation point saved as `best.pt`.

### Horizon 6,000

| step | lr | train loss | val loss | val ppl | grad mean | grad max | clipped | elapsed s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 9.974e-04 | 3.8858 | 3.5278 | 34.0500 | 0.965 | 3.075 | 0.224 | 11.1 |
| 500 | 9.872e-04 | 3.5119 | 3.4619 | 31.8783 | 0.819 | 1.752 | 0.024 | 22.5 |
| 750 | 9.694e-04 | 3.3181 | 3.0482 | 21.0776 | 0.848 | 1.179 | 0.024 | 34.1 |
| 1,000 | 9.443e-04 | 2.9730 | 2.8806 | 17.8253 | 0.889 | 1.475 | 0.032 | 45.7 |
| 1,250 | 9.123e-04 | 2.7675 | 2.7945 | 16.3541 | 0.884 | 1.424 | 0.032 | 57.3 |
| 1,500 | 8.741e-04 | 2.6643 | 2.6938 | 14.7874 | 0.853 | 1.170 | 0.016 | 68.8 |
| 1,750 | 8.302e-04 | 2.6132 | 2.6795 | 14.5776 | 0.851 | 1.085 | 0.016 | 80.4 |
| 2,000 | 7.814e-04 | 2.5456 | 2.6081 | 13.5728 | 0.853 | 1.343 | 0.020 | 92.0 |
| 2,250 | 7.286e-04 | 2.4959 | 2.6199 | 13.7349 | 0.863 | 1.170 | 0.024 | 103.5 |
| 2,500 | 6.727e-04 | 2.4521 | 2.6086 | 13.5802 | 0.868 | 1.103 | 0.020 | 114.6 |
| 2,750 | 6.147e-04 | 2.4024 | 2.5370 | 12.6422 | 0.867 | 1.293 | 0.012 | 125.6 |
| 3,000 | 5.556e-04 | 2.3318 | 2.5079 | 12.2787 | 0.873 | 1.410 | 0.008 | 137.2 |
| 3,250 | 4.963e-04 | 2.3059 | 2.5060 | 12.2560 | 0.902 | 1.108 | 0.032 | 148.8 |
| 3,500 | 4.380e-04 | 2.2421 | 2.4969 | 12.1443 | 0.899 | 1.441 | 0.028 | 160.3 |
| 3,750 | 3.817e-04 | 2.1813 | 2.4505 | 11.5947 | 0.903 | 1.089 | 0.040 | 171.8 |
| 4,000 | 3.282e-04 | 2.1160 | 2.4727 | 11.8539 | 0.932 | 1.985 | 0.072 | 183.3 |
| 4,250 | 2.786e-04 | 2.0689 | 2.4576 | 11.6766 | 0.953 | 1.415 | 0.152 | 194.3 |
| 4,500 | 2.338e-04 | 2.0035 | 2.4172 | 11.2148 | 0.981 | 1.204 | 0.304 | 205.2 |
| 4,750 * | 1.944e-04 | 1.9480 | 2.4032 | 11.0591 | 1.002 | 1.219 | 0.492 | 216.7 |
| 5,000 | 1.612e-04 | 1.8941 | 2.4223 | 11.2717 | 1.035 | 1.367 | 0.724 | 228.2 |
| 5,250 | 1.348e-04 | 1.8409 | 2.4345 | 11.4101 | 1.055 | 1.390 | 0.864 | 239.2 |
| 5,500 | 1.156e-04 | 1.8057 | 2.4069 | 11.0991 | 1.091 | 1.373 | 0.956 | 250.2 |
| 5,750 | 1.039e-04 | 1.7545 | 2.4531 | 11.6241 | 1.114 | 1.260 | 0.980 | 261.1 |
| 6,000 | 1.000e-04 | 1.7198 | 2.4487 | 11.5731 | 1.135 | 1.349 | 0.984 | 272.1 |

`*` marks the best subset-validation point saved as `best.pt`.

### Horizon 12,000

| step | lr | train loss | val loss | val ppl | grad mean | grad max | clipped | elapsed s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 9.996e-04 | 3.9489 | 3.5434 | 34.5836 | 0.978 | 2.507 | 0.268 | 11.1 |
| 500 | 9.974e-04 | 3.5208 | 3.4673 | 32.0511 | 0.835 | 1.778 | 0.032 | 22.5 |
| 750 | 9.933e-04 | 3.2393 | 2.9603 | 19.3039 | 0.874 | 1.268 | 0.020 | 34.1 |
| 1,000 | 9.872e-04 | 2.9140 | 2.8570 | 17.4094 | 0.901 | 1.225 | 0.032 | 45.7 |
| 1,250 | 9.792e-04 | 2.7459 | 2.7768 | 16.0672 | 0.882 | 1.235 | 0.020 | 57.3 |
| 1,500 | 9.694e-04 | 2.6602 | 2.7012 | 14.8982 | 0.857 | 1.157 | 0.016 | 68.9 |
| 1,750 | 9.577e-04 | 2.6196 | 2.7251 | 15.2584 | 0.860 | 1.270 | 0.012 | 80.5 |
| 2,000 | 9.443e-04 | 2.5588 | 2.5996 | 13.4582 | 0.846 | 1.149 | 0.016 | 91.6 |
| 2,250 | 9.291e-04 | 2.5165 | 2.6235 | 13.7832 | 0.849 | 1.012 | 0.004 | 103.1 |
| 2,500 | 9.123e-04 | 2.4719 | 2.6215 | 13.7566 | 0.842 | 1.010 | 0.008 | 114.3 |
| 2,750 | 8.939e-04 | 2.4380 | 2.5507 | 12.8164 | 0.842 | 1.367 | 0.012 | 125.3 |
| 3,000 | 8.741e-04 | 2.3871 | 2.5638 | 12.9853 | 0.839 | 1.175 | 0.032 | 136.9 |
| 3,250 | 8.528e-04 | 2.3696 | 2.5502 | 12.8091 | 0.847 | 1.357 | 0.016 | 148.0 |
| 3,500 | 8.302e-04 | 2.3116 | 2.4687 | 11.8074 | 0.815 | 0.981 | 0.000 | 159.6 |
| 3,750 | 8.064e-04 | 2.2674 | 2.5013 | 12.1978 | 0.806 | 1.051 | 0.008 | 171.2 |
| 4,000 | 7.814e-04 | 2.2238 | 2.5038 | 12.2286 | 0.805 | 1.053 | 0.004 | 182.3 |
| 4,250 | 7.555e-04 | 2.1873 | 2.5462 | 12.7587 | 0.801 | 1.036 | 0.004 | 193.3 |
| 4,500 | 7.286e-04 | 2.1416 | 2.4580 | 11.6814 | 0.806 | 1.020 | 0.004 | 204.4 |
| 4,750 | 7.010e-04 | 2.0991 | 2.4540 | 11.6342 | 0.821 | 1.210 | 0.024 | 216.0 |
| 5,000 | 6.727e-04 | 2.0601 | 2.4832 | 11.9800 | 0.825 | 1.317 | 0.004 | 227.6 |
| 5,250 | 6.439e-04 | 2.0126 | 2.4216 | 11.2640 | 0.836 | 1.142 | 0.028 | 238.7 |
| 5,500 * | 6.147e-04 | 1.9748 | 2.3813 | 10.8195 | 0.848 | 1.145 | 0.012 | 250.3 |
| 5,750 | 5.852e-04 | 1.9073 | 2.4061 | 11.0902 | 0.859 | 1.049 | 0.012 | 262.0 |
| 6,000 | 5.556e-04 | 1.8590 | 2.4514 | 11.6044 | 0.876 | 1.019 | 0.012 | 273.0 |
| 6,250 | 5.259e-04 | 1.7910 | 2.5103 | 12.3091 | 0.899 | 1.078 | 0.040 | 284.1 |
| 6,500 | 4.963e-04 | 1.7490 | 2.4846 | 11.9967 | 0.928 | 1.108 | 0.084 | 295.2 |
| 6,750 | 4.670e-04 | 1.6819 | 2.5050 | 12.2431 | 0.957 | 1.279 | 0.216 | 306.3 |
| 7,000 | 4.380e-04 | 1.6019 | 2.5867 | 13.2855 | 0.979 | 1.149 | 0.380 | 317.4 |
| 7,250 | 4.095e-04 | 1.5437 | 2.6450 | 14.0829 | 1.012 | 1.356 | 0.564 | 328.4 |
| 7,500 | 3.817e-04 | 1.4660 | 2.6586 | 14.2766 | 1.040 | 1.265 | 0.696 | 339.5 |
| 7,750 | 3.545e-04 | 1.3727 | 2.6793 | 14.5754 | 1.060 | 1.309 | 0.820 | 350.6 |
| 8,000 | 3.282e-04 | 1.2986 | 2.7331 | 15.3801 | 1.086 | 1.318 | 0.864 | 361.7 |
| 8,250 | 3.029e-04 | 1.2344 | 2.8045 | 16.5185 | 1.107 | 1.319 | 0.912 | 372.8 |
| 8,500 | 2.786e-04 | 1.1316 | 2.9187 | 18.5165 | 1.111 | 1.424 | 0.904 | 383.8 |
| 8,750 | 2.556e-04 | 1.0625 | 2.9214 | 18.5678 | 1.142 | 1.419 | 0.932 | 394.9 |
| 9,000 | 2.338e-04 | 1.0017 | 2.9705 | 19.5015 | 1.141 | 1.482 | 0.928 | 406.1 |
| 9,250 | 2.134e-04 | 0.9209 | 3.0533 | 21.1858 | 1.145 | 1.419 | 0.912 | 417.2 |
| 9,500 | 1.944e-04 | 0.8533 | 3.0992 | 22.1810 | 1.159 | 1.461 | 0.944 | 428.3 |
| 9,750 | 1.770e-04 | 0.7789 | 3.1904 | 24.2976 | 1.137 | 1.423 | 0.860 | 439.4 |
| 10,000 | 1.612e-04 | 0.7411 | 3.2002 | 24.5386 | 1.146 | 1.529 | 0.900 | 450.4 |
| 10,250 | 1.471e-04 | 0.7057 | 3.3065 | 27.2902 | 1.135 | 1.550 | 0.856 | 461.5 |
| 10,500 | 1.348e-04 | 0.6261 | 3.3629 | 28.8737 | 1.096 | 1.563 | 0.744 | 472.6 |
| 10,750 | 1.243e-04 | 0.5937 | 3.4270 | 30.7831 | 1.091 | 1.541 | 0.732 | 483.7 |
| 11,000 | 1.156e-04 | 0.5577 | 3.4691 | 32.1089 | 1.079 | 1.570 | 0.724 | 494.7 |
| 11,250 | 1.088e-04 | 0.5293 | 3.5308 | 34.1517 | 1.068 | 1.596 | 0.664 | 505.8 |
| 11,500 | 1.039e-04 | 0.4960 | 3.5811 | 35.9121 | 1.042 | 1.557 | 0.620 | 516.8 |
| 11,750 | 1.010e-04 | 0.4696 | 3.6145 | 37.1330 | 1.029 | 1.400 | 0.588 | 527.9 |
| 12,000 | 1.000e-04 | 0.4551 | 3.6582 | 38.7926 | 1.018 | 1.458 | 0.528 | 539.0 |

`*` marks the best subset-validation point saved as `best.pt`.

## 6. Generalization analysis

- **Horizon 3,000**: best full val 2.7804 at step 2,750 (91.7% of budget); train subset 2.4827; gap 0.2977; final full val 2.7666; deteriorated after best: False; best lr 1.156e-04.
- **Horizon 6,000**: best full val 2.7956 at step 4,750 (79.2% of budget); train subset 2.1682; gap 0.6274; final full val 2.8359; deteriorated after best: True; best lr 1.944e-04.
- **Horizon 12,000**: best full val 2.7911 at step 5,500 (45.8% of budget); train subset 2.0666; gap 0.7245; final full val 4.1301; deteriorated after best: True; best lr 6.147e-04.

Measured: lower training loss does not imply better generalization. The train/validation gap widens after the early optimum in every run where the final full-val loss exceeds the best full-val loss.

## 7. Generated text

Generation settings: {"max_new_tokens": 40, "temperature": 0.8, "top_k": null}.

### initial

- Prompt 'It is a truth universally true that' -> It is a truth universally true thather bethbut Eliza, and ed 9de‘ou.”7d be 3lizakDXat be Elizabethll dalceosit sed y been1x2liz; and 
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudiceher bethbut en , and ch 9P‘ou.”7d x43lizbethld Xat be Elizabethll dirceosit sed y urenXx2liz; and 
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyDbethbut Eliza, and edwith y‘ou.”twaha3lizbethld Xat be Elizabethll dby ceout sed y urshe Xx2liz; and 
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost andher bethbut Eliza, and ed 9de‘ou.”7d beha3lizakDXhabe Elizabethll dalceosit sed y been1x2liz; and 
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughher bethbut Eliza, and ed9P‘ai.”twvery  3lizbethld Xto be Elizabethll dirceosit sed y Ren1but 2liz; and 

### h3000_best

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
face to permistress wish of those to be ill mannumstance. Her proposal
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice, though to permistress wish though Elizabeth would have doubtless those repetime; and 
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyright. I am
quit the two much think way there were other temper, I wish; and 
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
face to permistress of the more than Elizabeth would have doubtless of the
repetime; and 
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughing to accused her. ‘With she
reguards and Elizabeth had almoved at Longbourn with hi

### h3000_final

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
face to permistress which more than Elizabeth and ill make opinion of the regard with whi
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice, that
to do much as to the same time and happiesday followed Lydia house and 
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyrious, I
myself.”

They were sat distance of a mother, at Longbourn with an 
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
face to permonly as she would. As Elizabeth would have more other take the
reproper; and 
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughing to the objectionable two manners and happied, and a follow my prepetime; and 

### h6000_best

- Prompt 'It is a truth universally true that' -> It is a truth universally true that, I shall go on ‘With pride, then shall be mentify the truptable moment. 
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice, though I
myself.”

The might instants of the summerts of their proposal
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyriticoldren? That shall we are to be ill my follow my preparticu
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
face of the most understood regure. It is more follow so great proposal
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughters. I
was most business, I cannot ashamed there are opinion to be
interferen

### h6000_final

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
face to perhaps.”

They repared to the rest, and then no objection
of Wic
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice, though
always love, more than Elizabeth and the matter opinion of the real peo
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyright on your
my represent price of your uncivility?”

“I never say,”
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
face of the most conscious regue. His regrets were town the requestion; and 
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughters. I
will this---would! I shall be married to be take the
repetimat

### h12000_best

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
face to get it as the twenty which is not ashame to doubts of
Lydia is women very 
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudice,
and the objectional twenty which was no unwelcomed by politened to be
proprie
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copy,
reserved for myself as to wait which I can happily do not open so furn in persu
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
resentment which musted twenty regular figures opinion of the real plain
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughing face with which miable to wahaevery
assurance, and he followed her repeated per; and 

### h12000_final

- Prompt 'It is a truth universally true that' -> It is a truth universally true that
     I shall be quite determined to him, to conceal them, I tell you
     in Hert
- Prompt 'The Project Gutenberg eBook of Pride and Prejudice' -> The Project Gutenberg eBook of Pride and Prejudices Austen,
with the utmost place weeks, the day of every titute, and the evening 
- Prompt 'You have the right to keep your own copy' -> You have the right to keep your own copyriad, and I
ask you look you no.”

“I think you would liked to like it which you tell me 
- Prompt 'most other parts of the world at no cost and' -> most other parts of the world at no cost and
serves, and that his country.

Elizabeth had now observed that Jane had given so so
- Prompt 'xyzzy quux blorf plugh' -> xyzzy quux blorf plughing her spirits; and ‘Mrs.
Bingley was so coming, in dance, deigned. Hurst, though

Interpretation: compare fluency, repetition, and degeneration across the initial model, each best checkpoint, and each final checkpoint. Lowest validation loss is not automatically the best-looking sample.

## 8. Recommendation

Measured: the predeclared subset-validation rule selected horizon 3,000 step 2,750 (`best.pt`) with full-val loss 2.7804 (ppl 16.1256). Among all saved checkpoints, horizon 3,000 step 3,000 (`final.pt`) has the lowest full-val loss, 2.7666 (ppl 15.9047).
The gap between the selected best checkpoint and that final checkpoint is only 0.0138 full-val loss, so this is a plateau near 3,000 steps rather than evidence that training longer is useful. Prefer horizon 3,000; either checkpoint is defensible, but `final.pt` has the better measured full-val loss.

Interpretation: if a shorter horizon matches or beats a longer one, the additional updates are not useful and mainly increase overfitting. If all horizons peak early and then deteriorate, the useful step budget is roughly the best-step range. Whether to change data, capacity, or regularization should be decided by a separate experiment; architecture, tokenizer, and corpus were intentionally held fixed here.

## 9. Full test-suite result

444 passed in 40.35s (full pytest suite; includes 6 new exp28 tests).
