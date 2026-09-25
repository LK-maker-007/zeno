# Ava blueprint

Date: 2026-09-25 (updated for plan v3). Part of the plan in `plan.md` (goal §1.1, rules §1.2, ledger §1.3).

Tags:
- **[V]** measured directly.
- **[C]** a primary source says it.
- **[C✓]** a quote I checked word for word in the downloaded source.
- **[H]** estimate or hypothesis, untested.
- **[R]** from memory, not checked against a source.

**Rule 1 (plan v3): no borrowed trained model inside Ava.** Every existing model in this file is an **opponent on the scoreboard**, not a part of Ava. Ava's organs are trained by us.

**No opponent below has been measured on this laptop.** No published benchmark exists for any of them on an i5-11300H or any Tiger Lake chip. Every speed for this machine is [H] until measured.

---

## 1. What Ava is meant to be

The target: a virtual human-like AI on this laptop (i5-11300H, 4 cores, 15 GB RAM, no discrete GPU), using webcam and microphone. You turn on the camera and talk to her. She:
- sees you and recognises you by name (enrolled, consenting people only);
- notices your facial movements, where you are looking, and whether you look tired;
- talks and listens in real time;
- remembers you and your life, and says so when she does not know something.

### How Ava grows (plan v3)

Ava is built from our own organs only, so she grows in stages:

| Stage | Ava can | Our organs |
|---|---|---|
| Ava 0 | Talk in text, in a small world, like a young child; remember what you tell her; say "I don't know" on near-misses | Child-mind (C1) + honest memory (Design D, M1) |
| Ava 1 | Keep learning about you without forgetting | + learning on the job (M2) |
| Ava 2 | Hear you and speak | + our ears, voice and turn-taking (I2, I5) |
| Ava 3 | See you, know you by name, notice your face | + our eyes (I3, I4) |

The camera-and-voice Ava in the list above arrives at Ava 3. Each organ joins only after it has beaten its opponents on the scoreboard (`plan.md` §7).

## 2. Architecture

```
 webcam ──► EYES ─────────────┐
            face, who, pose,  │
            gaze, movements   ▼
                         ┌──────────┐      ┌──────────────┐
 mic ────► EARS ───────► │  MIND    │ ◄──► │   MEMORY     │  (ours: Design D)
           speech→text   │ (core)   │      │ facts+sources│
                ▲        └────┬─────┘      │ "I don't     │
                │             │            │  know" signal│
          TURN-TAKING ◄───────┤            └──────────────┘
          (when to speak)     ▼
                            VOICE ──► speaker
```

## 3. Opponents on the scoreboard

The strongest CPU-capable existing model per organ. Our organ for each slot must beat it (at equal or lower cost) before it joins Ava. Licences matter only if an opponent's weights are redistributed; they are never shipped inside Ava.

| Organ | Opponent | Its numbers | Licence | Known weakness (our opening) |
|---|---|---|---|---|
| Face detection | **YuNet** (OpenCV) | 0.69 ms at 160×120 on an i7-12700K [C] | MIT [C] | Faces only ~10–300 px; downscale the frame [C] |
| Face landmarks, pose, blinks, gaze | **MediaPipe Face Landmarker** | 478 landmarks + 52 blendshapes [C] | Apache-2.0 [C] | No published desktop CPU speed; blendshape error 0.199 vs spread 0.237 on 511 lab samples [C] |
| Who is this | **SFace** (OpenCV) | 5.09 ms per crop on i7-12700K; LFW 0.9940 [C✓] | Apache-2.0 [C] | Only LFW published; no demographic breakdown |
| Facial movements | **OpenCV zoo FER** (optional) | 1.79 ms per crop; RAF-DB 88.27% [C] | Apache-2.0 [C] | Seven fixed emotion labels. We use its output as *movements*, not emotions (§6). |
| Speech detection | **Silero VAD v6** | <1 ms per 30 ms chunk on one CPU thread [C] | MIT [C] | Detects speech, not end of turn |
| End of turn | **Smart Turn v3.x** | 8 MB; 12.6 to 94.8 ms on cloud CPUs; English 94.31% [C] | BSD-2 [C] | Misjudges short replies ("yes", "okay") |
| Ears (speech to text) | **Moonshine v2 Small** | 148 ms on Apple M3; average WER 7.84% [C] | MIT [C] | English only; no x86 CPU numbers |
| Mind | **Qwen3-0.6B / 1.7B**, Q4_K_M; plus a same-size Transformer trained by us on the same small world | MMLU-Redux 44.6 / 64.4 [C]; ≤50 / ≤18 tokens/s ceiling at 20 GB/s [H] | Apache-2.0 [R] | Small models barely know facts (SimpleQA 2.2 for Gemma 3 1B) [C] and do not say so |
| Voice | **Kyutai Pocket TTS** (Kokoro as alternative) | ~200 ms to first audio on M4; ~2.3–2.5× real time on a 4-vCPU x86 VM [C] | Licence conflict: HF card CC-BY-4.0, a secondary source says MIT; resolve before sharing | Cannot insert pauses; no emotion control |
| Memory | GRACE-style codebook, LMLM-style threshold, SMF, RAG text store, plain fine-tuning (`plan.md` §1.3) | See `plan.md` §3 | various | Near-misses, forgetting, no calibrated "I don't know" |
| Scene questions ("what am I holding?") | **SmolVLM2-256M** via OpenVINO, on request only | 0.48 s end to end on a Core Ultra 7 265K [C] | Apache-2.0 [C] | Expect seconds on this laptop [H] |

**Opponents with research-only licences** (fine to benchmark against, never to ship): InsightFace, OpenFace 3, LibreFace and EdgeFace (Hugging Face) weights [C]. Piper is GPL-3.0 [C].

## 4. Speed targets on this laptop [H]

These are the opponents' estimated speeds. Our organs must match or beat them.

**Eyes (one CPU core, 10 to 15 fps):** YuNet 3–8 ms + landmarks 10–25 ms + derived cues ~1 ms per frame. The face embedding runs only when a new face appears, then every 1–2 s. About 15 to 35 ms per frame in total.

**Conversation (end of your speech to Ava's first sound):**

| Stage | Estimate |
|---|---|
| Deciding you have finished | 300–600 ms |
| Final transcription (mostly overlapped) | 0–250 ms |
| Mind: first words | 500–900 ms |
| Voice: first sound | 200–400 ms |
| **Total** | **~1.0 to 2.0 s** |

Reference points [C]:
- Across 10 languages, gaps between turns have a **mode of about 200 ms**, but producing speech takes over 600 ms. People predict the end of a turn instead of waiting for it (Levinson & Torreira 2015).
- The best measured local stack, on a far faster Apple M4 Max, took 1.4 to 1.6 s.

**All organs share 4 cores and about 20 GB/s of memory bandwidth.** Running together will be slower than each alone [H].

## 5. Where Ava invents

From the failure lists in the research. Ranked by value to Ava × cost on this laptop.

| # | Invention | Failure it fixes (evidence) | Opponent to beat |
|---|---|---|---|
| **I1** | **Honest memory (Design D)** | Small minds do not know facts and do not say so; assistants overwrite and miss facts (LongMemEval: ChatGPT "tended to overwrite crucial information") [C✓] | Memory organ (new) |
| **I2** | **Turn-taking that sees and predicts, on CPU** | Waiting 300–600 ms of silence, still ~1 false cutoff in 10 at 300 ms [C]; noise drops audio-only turn prediction from 84% to 52%, video brings it back to 72% [C]; no shipped detector uses a camera [H, limited search]; prediction work exists but not on a 4-core CPU [C] | Silero + Smart Turn |
| **I3** | **Personal-baseline reading of faces** | Emotion labels sit near human agreement (AffectNet annotators "agreed on 60.7%" [C✓]); faces map "many-to-many" to emotions (Barrett [C✓]). Better target: changes against *your own* usual face, combined with voice and context. | FER model |
| **I4** | **Never name a stranger as a friend** | Face recognition false positives "vary by factors of 10 to beyond 100 times" across groups (NIST [C✓]); compact models lose most on hard groups (East Asian 51.03 for buffalo_s [C]). Per-person thresholds calibrated on your own camera and room. | SFace threshold |
| **I5** | **Silence that stays silent** | Whisper-family ASR invents phrases during pauses (~1% of transcripts) [C]; laptop speakers feed back into the mic [C] | Ears / echo handling |

**Order:** I1 first, together with the child-mind (Ava 0). I2 next, when Ava gets ears and a voice (Ava 2). Both fix real, measured failures, are cheap on CPU, and are what makes Ava feel like a person: she remembers you honestly, and she answers on time.

## 6. What Ava may claim to perceive

| She can say | She must not say |
|---|---|
| "I see you." | "You are sad / angry / lying." |
| "I think this is Priya" (with confidence, enrolled people only) | Names for anyone not enrolled |
| "You're frowning more than usual. Is something off?" | Claims about intentions or sincerity |
| "You look tired" (blinks, head pose) | Certainty about inner feelings |

Why:
- Barrett et al. 2019: "It is not possible to confidently infer happiness from just a smile, anger from a scowl, or sadness from a frown"; such technology "detects facial movements, not emotional expressions" [C✓].
- The EU AI Act agrees. Emotion recognition "does not include physical states, such as pain or fatigue", nor the mere detection of "a frown or a smile" unless used to infer emotions (Recital 18) [C✓].

This matches the project's honesty principle: Ava reports what she sees, not what she guesses.

## 7. Consent and law

- **Enrol a face only with the person's explicit agreement.** Store face templates on the laptop only. Never identify strangers.
- **EU:** a natural person using AI "in the course of a purely personal non-professional activity" is outside the AI Act's deployer obligations (Art. 2(10)) [C✓]. Distributing Ava to others is different. Emotion inference is banned in workplaces and schools (Art. 5(1)(f)) [C]. GDPR requires explicit consent for biometric identification (Art. 9) [C].
- **Illinois BIPA:** "Private entity" means "any individual…" [C✓]. No personal-use exception was found. Written release and a retention policy are required; damages are $1,000 to $5,000 per violation [C].
- **Not legal advice.** Check before sharing Ava with anyone.

## 8. Build order

Follows `plan.md` §7.

| Step | What | Hardware | Done when |
|---|---|---|---|
| SB | **Scoreboard**: small-world data generator, tests T1 to T5, opponents registered | Laptop | Every opponent runs end to end with recorded numbers |
| B0 | **Measure the opponents on this laptop** (latency, CPU, RAM), idle machine | Laptop | Measured numbers replace the [H] estimates above. No published Tiger Lake numbers exist, so this is useful on its own. |
| C1 | Our child-mind, trained from scratch | Kaggle | Matches the same-size opponent on small-world language (T4) |
| M1 | Our honest memory (Design D) attached | Laptop + Kaggle | Passes its confirmatory test: **Ava 0** |
| M2 | Learning on the job | Laptop + Kaggle | Passes its kill test: **Ava 1** |
| I2, I5 | Design, red-team and build our ears, voice, turn-taking | Laptop + Kaggle | Beat their opponents: **Ava 2** |
| I3, I4 | Our eyes | Laptop + Kaggle | Beat their opponents: **Ava 3** |

**Progress (2026-09-25):** SB is built and running. C1 is built and tested locally (gate G1 passed), and its full training waits on a Kaggle run. See `plan.md` §7 and §11.

B0 needs an idle machine.

## 9. Unverified items

- Every speed on this laptop.
- Pocket TTS licence (conflicting sources).
- Qwen3 licence [R] (matters only for benchmarking).
- MediaPipe desktop CPU speed.
- Moonshine on x86.
- Whether any shipped system uses a camera for turn-taking (limited search).
- Current status of the GDPR biometric-verification amendment.
- Whether BIPA reaches a purely private hobby app (no case law checked).

Voice-side figures (Smart Turn, LiveKit, Levinson & Torreira, Whisper hallucination rates) were read from web pages and are [C], not re-checked against saved copies.
