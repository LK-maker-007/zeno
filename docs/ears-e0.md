# E0: Ava's first ears, a known architecture trained from scratch

Date: 2026-09-26. Written before any E0 run. Part of `plan.md` (ledger §1.3, row "Ears") and `ava-blueprint.md` (§3 opponents, §5 I5).

Tags:
- **[V]** measured or read directly for this document.
- **[C]** a primary source says it; the link is given.
- **[H]** hypothesis or arithmetic, untested.
- **[R]** from memory, unchecked.

---

## 1. Why E0 exists

- **Rule 1** (no borrowed trained model inside Ava) means Ava's ears are trained by us. **Rule 2** allows a known architecture until ours beats it.
- E0 is to the ears what C1 is to the mind: a known architecture, trained from scratch. It has three jobs:
  1. prove the pipeline end to end: audio, features, training, decoding, scoring;
  2. become the same-size baseline that our own ears architecture must beat;
  3. race, on this laptop, against the ears opponent in `ava-blueprint.md` §3 (Moonshine v2). That race is a separate step.

## 2. Data

- **LibriSpeech**, "License: CC BY 4.0" [V, openslr.org/12]. Read English audiobook speech at 16 kHz.
- **Kaggle mirror:** `victorling/librispeech-clean`.
  - E0 decodes every file it uses. A missing audio file, a character outside the 28 symbols, or an empty split stops the run.
  - The run prints utterances, speakers and hours for each split, which records what the mirror held.
  - A separate CPU probe (`kaggle/librispeech-probe/`) cross-checks the mirror's hours against LibriSpeech's own `CHAPTERS.TXT`. It had not finished when E0 launched; its report is recorded when it does.
- **Splits:** train on `train-clean-100`; report `dev-clean` during training; score `test-clean` once, at the end.
- **Speakers:** `ears.train` refuses to run if any dev or test speaker appears in the training data, because a shared voice would inflate every score.
- **Only clean read speech:** see §7.

## 3. Setup

| Part | E0 | QuartzNet paper | Why the difference |
|---|---|---|---|
| Model | QuartzNet-5x5 per Table 1; 6,713,181 parameters [V, `tests/test_ears.py`] | 6.7M [C] | none |
| Dilation 2 | on C2 | the text says C4 | C4 is a 1×1 layer, where dilation does nothing [H] |
| Features | 64 log-mel bins, 25 ms Hann window, 10 ms hop, normalised per utterance and channel | not given in the paper text [V, pdftotext] | our choice |
| Characters | 28 plus the CTC blank | CTC loss [C]; the text read does not name the label set | none |
| Optimiser | AdamW, lr 2e-3, weight decay 1e-3, 1000 warm-up steps, cosine decay | NovoGrad [C] | AdamW is in torch, NovoGrad is not |
| Augmentation | 2 frequency masks (≤15 bins) and 2 time masks (≤50 frames, ≤ a fifth of the utterance) | speed perturbation and SpecCutout [C] | features are cached in RAM, so speed perturbation would mean re-decoding the audio |
| Data and epochs | 100 h, 40 epochs, stopped at 9 h of training | LibriSpeech, 300 epochs [C, Table 2]; all 960 h is implied, not stated in the text read [H] | Kaggle budget (§5) |
| Precision | fp16 autocast, CTC loss in fp32 | mixed precision [C] | T4 has no bf16 |
| Seeds | 0 and 1 | not stated | one seed is not a result |

QuartzNet paper: Kriman et al. 2020, https://arxiv.org/abs/1910.10261.

## 4. What the paper reached, and why E0 is not a reproduction

- QuartzNet-5x5 reached greedy WER 5.39 on dev-clean and 15.69 on dev-other after 300 epochs on LibriSpeech [C, Table 2], presumably all 960 hours [H].
- If so, E0 trains on about a ninth of that data, for under a seventh of the epochs, without speed perturbation.
- The paper reports no 100-hour result. E0's number is therefore measured, not predicted, and it may be far worse. That is acceptable: E0 is a baseline, not a claim.

## 5. Budget [H]

- **Compute:** 6.7M parameters at 50 output frames a second is about 6.7e8 FLOP per second of audio forward, and about 2e9 to train.
- **Time per epoch:** for the 15x5 model, "The training of the 15x5 model for 400 epochs took ≈ 5 days on one DGX1 server with 8 Tesla V100 GPUs" [C]. Assuming that was on 960 h, scaling by parameter count gives about 0.9 V100-hours per 960-hour epoch for 5x5, so about 5 minutes per 100-hour epoch on a V100.
- **On a T4:** a T4 is roughly a half to a third of a V100 on fp16 convolutions [R]. That gives 10 to 15 minutes per epoch, so 40 epochs take 7 to 10 hours. The 9-hour guard may stop the run early; it records how many epochs finished.
- **Quota:** one Kaggle session of up to about 11 hours, out of the 30-hour week.

## 6. Gates and kill (pre-registered)

- **G1, overfit sanity:** 16 training utterances, 500 steps, greedy CER ≤ 0.05 on those same utterances.
  - Fail: the pipeline is broken and E0's numbers are void.
  - The laptop version (8 utterances) reached CER 0.000 by step 100 [V, `runs/ears-overfit-local/`, not committed]. It also showed that batch-norm running statistics lag early on: CER was 0.965 at step 25 while training loss was already 0.014. That is why G1 gets 500 steps.
- **G2, learning sanity:** dev-clean greedy WER ≤ 0.50 for both seeds.
  - Fail: stop all ears work and debug.
  - This is a pipeline check, not a quality claim.
- **Reported, not gated:**
  - dev-clean and test-clean WER per seed, each with a 95% CI from a speaker bootstrap;
  - CER;
  - the mean and spread over the two seeds;
  - epochs completed.
- **Next, separately:** real-time factor on this laptop's CPU for E0 and for Moonshine v2 on the same test set; then the I5 silence test from `ava-blueprint.md` §5.

## 7. What E0 cannot show

- **Only clean read speech:** no noisy rooms, accents, conversation, or the laptop microphone Ava will hear through. E0 does not use `test-other`.
- **Greedy decoding only:** no language model. Every score is the acoustic model alone.
- **Two seeds** rule out one unlucky seed and nothing more.
