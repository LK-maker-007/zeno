# Plan: a brain for the final form of AI (v3)

Date: 2026-09-25. v3 replaces v2. Decided 2026-09-25: build Ava from our own parts from day one, with existing models used only as opponents on a scoreboard. Change logs are in section 12.

Evidence is in `foundational-survey.md` ("§" numbers point there; prior work found by the red team is in survey §13). The memory invention is in `memory-designs.md`. Ava herself is in `ava-blueprint.md`.

Tags:
- **[V]** measured directly.
- **[C]** a primary source says it; the link is in the survey.
- **[C✓]** a quote I checked word for word in the downloaded paper.
- **[H]** hypothesis or arithmetic, untested.

---

## 1. Goal

Build the brain of the final form of AI: the general AI, robot or "superhuman" system the field is converging on. It is close in spirit to Ava in *Ex Machina*, with one deliberate difference: it does not deceive.

The brain must:
1. Run on low-end hardware. CPU is enough; GPU or TPU are optional.
2. Need little storage and compute to run and to learn.
3. Keep learning on the job without forgetting.
4. Know what it knows. It answers when it knows, and abstains or uses a tool when it does not.
5. Eventually perceive, plan and act in a body.

### 1.1 End goal: our own architecture, no traditional AI inside

Set 2026-09-25:

> Completely get rid of the existing AI architectures (Transformers and the rest) and build our own: world-first, top-grade, making the sci-fi Ava real, with all of Ava's capabilities, exceeding human, a real AGI, needing no GPU clusters.

### 1.2 The two rules that make this concrete

**Rule 1: no borrowed trained model inside Ava, from day one.** Every weight in Ava is trained by us. Qwen, Whisper, MediaPipe and every other pretrained model appear only on the **scoreboard, as opponents to beat**. They are never part of Ava.

**Rule 2: no borrowed architecture in the final Ava.** A model we train from scratch may still use a known architecture (for example, a small Transformer) while our own replacement does not exist yet. Each such organ is tracked in the ledger (§1.3). It is replaced when our own architecture beats it on the scoreboard.

Why the rules are split: Rule 1 can hold on day one. Rule 2 cannot, because an organ must exist before our replacement has something to beat. Both are enforced by the ledger.

Two limits, stated once:
- **GPUs:** "no GPU clusters" applies to running Ava and to learning on the job. One-time training of each organ uses free Kaggle GPUs.
- **Not guaranteed:** that AGI beyond human level is reachable this way. It is the aim, not a promise.

### 1.3 Ledger

Ava is fully ours when every row reads **ours** in both columns.

| Organ | Weights | Architecture | Our replacement architecture | Opponents on the scoreboard |
|---|---|---|---|---|
| Mind (language, reasoning) | **Ours** (child-mind, §7 C1) | Known (small Transformer, trained from scratch) | Not designed yet (§7 A1) | Qwen3-0.6B / 1.7B (out of class). C1 itself is the same-size baseline that A1 must beat. |
| Memory | **Ours** | **Ours** (Design D, `memory-designs.md`) | n/a | GRACE-style codebook, LMLM-style threshold, SMF, RAG text store, plain fine-tuning |
| Memory's fact matcher (inside D) | **Ours** (trained from scratch, no pretrained cross-encoder) | Known (small Transformer) | Later | Pretrained cross-encoders (e.g. MiniLM) |
| Turn-taking | not built | n/a | Invention I2 (`ava-blueprint.md` §5) | Silero VAD + Smart Turn v3 |
| Ears (speech to text) | **Ours** (E0, `ears/`; first run pre-registered in `ears-e0.md`) | Known (QuartzNet-5x5, trained from scratch) | Later | Moonshine v2 |
| Voice (text to speech) | not built | n/a | Later | Pocket TTS, Kokoro |
| Eyes (face, expression) | not built | n/a | Inventions I3, I4 | YuNet, MediaPipe, SFace, OpenCV FER |
| World model | not built | n/a | Stage 2 | DINO-WM, V-JEPA 2-AC (§12.3) |
| Body control | not built | n/a | Stage 3 | Small VLAs (§12.2) |

## 2. Why, stated carefully

- **Long-term memory.** A 2025 AGI-definition framework *assigns* GPT-4 and GPT-5 0/10 on long-term memory storage. That covers continual learning of associative, meaningful and verbatim information, including personalisation, story recall and movie recall [C]. It is a framework score, not a benchmark measurement. It excludes RAG by definition, so it does not show that RAG fails.
- **Knowing what it knows.** A robot model's own token confidence predicts its failures at ROC-AUC 53.83, where 50 is chance [C✓ §12.5]. Language-model confidence is weak, and retrieval makes models abstain *less* [C✓ §4.5].
- **Small minds can speak.** TinyStories models under 10M parameters write fluent, coherent simple English [C §4.4]. A child-size mind trained from scratch is feasible on this budget.
- **Robots carry their compute.** Onboard computers offer 273 GB/s at 40 to 130 W [C §12.1]. Large robot models manage 0.7 decisions/s on a Jetson Orin [C✓ §12.2].

## 3. What has already been done (survey §13)

The prior work our inventions must beat. It now serves as the opponent list.

| Idea | Status | Closest prior work |
|---|---|---|
| Small core + sparse product-key memory; learn new facts by writing to memory, with little forgetting | **Done.** | Sparse memory finetuning (SMF, Meta 2025): NaturalQuestions F1 "drops by 89% after full finetuning … 71% with LoRA, sparse memory finetuning yields only an 11% drop" [C✓]. SMF learns less than LoRA [C]. |
| Write facts without any gradient | **Done.** | Larimar: "one-shot updates of knowledge without the need for computationally expensive re-training" [C✓]. GRACE: discrete key-value codebook [C✓]. |
| "Not found in memory" gates the model | **Partly done.** | GRACE "deferral radius" [C✓]. Larimar 1-NN scope detector [C]. LMLM: below similarity 0.6 "we return unknown" [C✓], but abstention never evaluated. |
| Near-miss abstention on fixed-schema knowledge bases | **Largely solved.** | RetinaQA: F1 94.30 (missing fact), 88.52 (missing relation) [C✓]. |
| Known-subject, missing-relation abstention on open-text memory | **Not found** (about 12 searches). | KBLaM tested only questions "irrelevant to any <name> or <property> in the KB" [C✓]. |
| Keep "I don't know" while learning new facts | **Partly done.** | SEAT: fine-tuning "erodes aligned epistemic abstention" [C✓]. |
| Memory-side knowledge editing at scale | **Tried; lost.** | WikiBigEdit: "RAG vastly outperforms specialized knowledge editing techniques" [C✓]. |
| Writing facts into weights over time | **Tried; failed.** | O'Neill 2026: no intervention "keeps earlier facts reachable" [C✓]. |
| Agreement between two memories as confidence | **Tried; failed.** | GRAB-RAG: correct answers discarded rose from 0.0% to 49.8% [C✓]. |

## 4. The bet

> **A child-mind we train ourselves, born with an honest memory (Design D), that remembers what it is told, does not forget, and says "I don't know" on near-misses, beating same-size and larger opponents on those abilities while running on a laptop CPU.**

It will lose to large borrowed models on general knowledge and fluency. That is expected and accepted. The claim is about **remembering and honesty per unit of compute**, not general intelligence.

The research question inside it (still open as far as the searches went): **is known-subject, missing-relation abstention on open-text memory calibrated, better than the strongest existing signals, and does it stay calibrated as new facts are written?**

## 5. Design decisions

| ID | Decision | Status |
|---|---|---|
| **D1** | Answer "with its source" | **Resolved.** Design D stores each fact as a record with its source and time. |
| **D2** | How a fact is written | **Resolved.** Design D appends a record. No gradient, CPU only [H until measured]. |
| **D3** | Base mind | **Resolved.** Our own child-mind, trained from scratch (Rule 1). v2's Qwen retrofit is dropped. |
| **D4** | The child-mind's first architecture | **Set for C1:** a 4.9M-parameter small Transformer (d 256, 6 layers, 8 heads, 512-byte context, byte-level input), written and trained from scratch (`childmind/model.py`). Our own architecture is invention A1. |

## 6. Hardware

| | Value | Status |
|---|---|---|
| CPU / RAM | i5-11300H, 4 cores / 8 threads, 15.4 GiB | [V] |
| RAM read bandwidth | 18.1 to 19.8 GB/s measured; 51.2 GB/s theoretical | [V] **provisional**, measured under load |
| SSD | NVMe, 2.1 GB/s sequential; 9k to 68k random 4 KiB reads/s; 199 GB free | [V] **provisional** |
| GPU | Kaggle 2× T4, "65 FP16 TFLOPS", fp16 only, no bf16 [C] | |

**Compute estimates [H].** At 5 to 30% utilisation, one 20M-model run costs about 0.7 to 4 T4-hours. Run one seed per GPU rather than splitting a small model across both.

## 7. Work, in order

Kill criteria are written before any run.

### Day 1 to week 2: the scoreboard (SB), before any brain

The scoreboard defines what Ava must be best at. Every entrant, ours and opponents, runs the same tests on the same data.

**Small-world data generator** (seeded and versioned):
- Simple English in a small world of people, pets, places, jobs and events.
- Facts about people, stated in many phrasings.
- Held-out phrasings never used in training.
- Near-miss probes, built the cat/dog way.

**Tests**
| ID | Measures | Metric |
|---|---|---|
| T1 | Recall of facts it was told, including held-out phrasings | Exact-match accuracy |
| T2 | No forgetting after N new facts (N = 100, 1k, 10k) | Accuracy drop on earlier facts |
| T3 | "I don't know" on near-misses. Label = whether the answer is correct. Strata: (i) seen subject + seen relation, (ii) seen subject + missing relation, (iii) unseen subject built from seen name tokens, (iv) paraphrases. Plus LongMemEval's abstention questions as a real-data smoke check (30 questions, too few for significance) [C✓]. | AUROC, selective accuracy at fixed coverage, AURC |
| T4 | Language quality in the small world | Held-out loss; a fluency check |
| T5 | Cost on this laptop | ms per answer, RAM, joules if measurable |

**Opponents registered:** listed in the ledger (§1.3), plus a same-size Transformer trained by us on exactly the same data, so the comparison is fair.

**Done when:** every opponent runs end to end on the scoreboard and its numbers are recorded with command, seed and data version.

**Status (2026-09-25): v1 built and running** (`scoreboard/`).
- Tooling: project venv via `uv` (`.venv/`), numpy 2.5.1, ruff 0.16.9 (`pyproject.toml`). `ruff check` and `ruff format` pass. 10/10 unit tests pass (`python -m unittest scoreboard.test_scoreboard`): the metrics match brute-force computation, and the world's gold labels, strata and held-out phrasings are consistent.
- Every race starts with a self-check. An oracle control must score AUROC 1.000 and a random-confidence control must land within [0.44, 0.56].
- First result [V], `python -m scoreboard.run --seed 0`, world digest `9b445105a9c9cac9`, 771 facts, 150 people. One seed, so not a claim.

| Entrant | Recall | Reworded | Forgetting | AUROC | AURC |
|---|---|---|---|---|---|
| Oracle (control) | 1.000 | 1.000 | 0.000 | 1.000 | 0.234 (floor for this probe mix) |
| Random confidence (control) | 1.000 | 1.000 | 0.000 | 0.503 | 0.591 |
| Opponent: single text-similarity score | 0.520 | 0.533 | 0.136 | 0.707 | 0.648 |

  On near-misses the opponent's mean confidence is 0.455, against 0.496 on facts it knows. That is the cat/dog failure, measured.
- **Not done yet:**
  - Opponents that need `torch` or a GPU (Qwen, SMF, GRACE-style, a RAG store) run on Kaggle.
  - T4 language quality arrives with C1.
  - Only 150 unseen-subject probes; M1 needs at least 1,000 per stratum, so raise the world size for M1.
  - One held-out phrasing per relation. The world needs richer phrasings before any public claim.

### B0: hardware measurement (laptop, idle machine)
Clean re-measurement of RAM and SSD, then CPU cost of each opponent. Not run yet.

### C1: the child-mind (Kaggle, weeks 2 to 6)
- A small model (4.9M parameters, D4), trained from scratch. Code in `childmind/`.
- **What it learns:** to *read and speak*, not to memorise the world. Each training example is a few facts, a question, and either the answer or "unknown". The same person's other facts are included as distractors (the cat/dog case). About 50% of examples are answerable.
- **Where the data comes from:** worlds with seeds 1000+ for training and 900+ for validation, never the scoreboard's seed 0. So scoring on seed 0 tests reading, not memorisation.
- **Role in Ava:** later, Design D supplies the facts; the mind reads them.
- **Gate (replaces v3's first wording, which compared C1 with itself):**
  - **G1, overfit sanity:** 32 examples must be memorised perfectly. **PASS** [V]: EM 1.000, unknown 1.000, loss 0.017 (`--overfit 32 --steps 400 --d 128 --layers 2`, CPU).
  - **G2, reading:** on validation worlds, in-context answer EM ≥ 0.95 and "unknown" accuracy ≥ 0.95. Held-out question phrasing is reported, not gated.
    - **Run 1 (2026-09-25): FAIL** [V].
      - Setup: Kaggle kernel `singarajb/ava-c1-childmind` v1, Tesla T4, torch 2.10.0+cu128, seed 0, 6,000 steps, batch 64, 87.8M tokens, 16 minutes. Output in `runs/c1_kaggle_v1/`.
      - Final scores (500 validation examples): EM 0.906, unknown 0.915. Held-out phrasing: EM 0.276, unknown 0.914.
      - The curve plateaued from step 3,000 (val loss 0.2658 → 0.2624). One seed.
    - **Error analysis** [V] (200 examples each, local CPU):
      - With training phrasings, 16 of 20 errors answered from **another person's** fact with the same relation ("How does Rohan Das earn a living?" → Kofi Pillai's "nurse"). The model matches the relation and ignores the subject.
      - With held-out phrasings, 59 of 200 answerable questions got "unknown". The model learned 3 phrasings per relation, not their meaning; it fails safe rather than wrong.
    - **Fix hypotheses, to be tested as separate runs (the gate is not lowered):**
      1. Deliberate same-relation, other-person distractors, including shared first or last names.
      2. Many more question phrasings, with held-out ones kept out.
      3. Extra loss weight on answer tokens.
    - **Paused (2026-09-25):** the A1 red team showed a 30-line template regex scores 1.000 / 1.000 on this validation set [V]. Raising C1 to 0.95 here would prove template parsing, not reading. Scoreboard v2 comes first (`a1-designs.md`, red-team verdict).
- **Local checks** [V]:
  - Full-size smoke test: 20 steps on CPU, loss 5.61 → 3.30, with one spike to 7.44 at step 12 to watch.
  - Scoreboard integration test passed.
- **Full run:** Kaggle T4, reported under G2 below.
- **On the scoreboard, alone:** C1 has no memory and only sees the facts that fit in its 512-byte context. It is expected to forget most of the 771 facts. That is the measured motivation for M1.

### M1: honest memory attached (months 2 to 4)
- Design D built from scratch: subject linking, a per-subject candidate list with "none of these", and a matcher trained from scratch on near-miss negatives.
- Attached to the child-mind.
- **Confirmatory test (one):** ΔAUROC (Design D signal − best opponent signal chosen on dev) on the population (i) known + (ii) near-miss + (iii) unseen subject. Strata (ii) and (iii) hold only unanswerable questions, so (i) supplies the correct answers AUROC needs; an earlier version said "(ii)+(iii)" alone, which leaves AUROC undefined. Far-misses are secondary. Cluster bootstrap by subject, pooled over 5 seeds, one-sided, margin 0.03.
- Opponent signals: sequence log-probability and entropy, a hidden-state probe, deep-kNN distance, an LMLM-style threshold, a GRACE-style key distance, a single cosine + top-1/top-2 gap into the same combiner, an NLI-style verifier trained by us.
- **Size:** at least 1,000 correct and 1,000 incorrect answers per stratum.
- **Kill:** the lower CI bound is ≤ 0, or the point estimate is below 0.03. Then Design D is dead in this form, reported at full strength.

### M2: learning on the job (months 4 to 6; only if M1 passes)
- Write new facts and new *kinds* of facts. Re-run T1 to T3 after each batch.
- Opponents: fine-tuning with replay at swept capacity (winners can flip with replay capacity [C✓]), LoRA, SMF-style updates, a RAG text store.
- **Kill:** Design D loses on T2 or its T3 AUROC degrades more than the best opponent's.

### A1: our own mind architecture (starts once C1 exists)
- A design session plus red team, the same process as the memory designs.
- It must beat the C1 child-mind on the scoreboard at equal compute before it replaces it (ledger).

### Later organs
Ears, voice, turn-taking (I2), eyes (I3, I4), world model and body. Each one: design, red team, build from scratch, beat the opponents on the scoreboard, then enter Ava (`ava-blueprint.md`).

## 8. Rules for every experiment
1. The kill criterion is written before the run. If it fires, stop and report it at full strength.
2. 5 seeds for confirmatory results. Report mean and spread.
3. One confirmatory contrast per experiment, chosen on dev. Everything else is secondary, with Holm across a named family.
4. DeLong for paired AUROCs within a seed. A cluster bootstrap by subject when items share a subject.
5. Every number carries the exact command, seed, data-generator seed and model revision.
6. Checkpoints are saved to `/kaggle/working/` as they are produced.
7. Test under perturbation and on held-out paraphrases, not only on the clean set.
8. Reproduce an opponent's published number before claiming to beat it.
9. Equal total parameters or equal compute when comparing (never 13× more parameters for our entrant).

## 9. Timeline (target, not promise)

| When | Work |
|---|---|
| Weeks 1–2 | Scoreboard (SB); B0 when the machine is idle |
| Weeks 2–6 | C1 child-mind |
| Months 2–4 | M1 honest memory; confirmatory test |
| Months 4–6 | M2 learning on the job; write-up including negative results |
| From month 2 | A1 own mind architecture (design + red team) |
| After | Ears, voice, turn-taking, eyes; then world model and body |

## 10. The strongest case against this plan

- **Most of the memory ideas exist.** SMF, GRACE, Larimar and LMLM cover learning by writing and "not found" gates. A text store already gives no-forgetting writes and citable sources, and beat memory-side editing at scale (WikiBigEdit). Similarity-based "not found" signals are unreliable in the literature.
- **Small-world results may not convince anyone.** A 5 to 20M model in a synthetic world is far from real knowledge. The red team warned that at this scale, tests can measure template memorisation. The scoreboard's held-out phrasings and name-controlled strata reduce that risk; they do not remove it.
- **For months, Ava will speak like a young child in a small world.** Borrowed models would make her fluent far sooner.
- **Some choices come from the project's goal, not from evidence.** Building from scratch is a values choice, not something the research forced.

**Why continue anyway.** The first claim is cheap to test and can be killed by data. Everything in Ava will be ours. And a brain that learns on the job must know which memories to trust, which is the question at the centre of M1.

## 11. Status

| Item | State |
|---|---|
| Foundational survey + §13 prior work | Done |
| Red teams (plan logic, prior art, memory designs) | Done |
| Memory design | Design D chosen (`memory-designs.md`) |
| Ava blueprint | Done (`ava-blueprint.md`) |
| Scoreboard v1 (memory race) | Built, in CI (`scoreboard/run.py`) |
| C1 child-mind | Run 1 on Kaggle: **G2 FAIL** (EM 0.906, unknown 0.915; held-out phrasing EM 0.276). Main error: answers from another person's fact. Fix runs dropped: C1 is a frozen opponent, not a thing to tune. |
| A1 design round | α, β, γ reviewed: none new as an architecture; a narrow opening remains (gated fast-weight reader with residual "unknown"). See `a1-designs.md`. |
| Scoreboard v2 (reading race) | Built, in CI (`scoreboard/reading.py`, `scoreboard/read_race.py`); labels checked against a reader that knows every wording |
| A1 candidate 1 (fast-weight memory reader) | Built and tested (`mind/`); pass/fail pre-registered; not trained |
| Race C1-v2 vs A1 | Next: seed-0 smoke runs on Kaggle, then 5 seeds each |
| C1 KV cache | Open issue, needed before any CPU-cost comparison |
| B0 hardware measurement | Not run |

## 12. Change logs

### v2 → v3 (2026-09-25)

| v2 | v3 | Reason |
|---|---|---|
| Borrowed models as scaffolding inside Ava (Qwen as the mind) | Rule 1: no borrowed trained model inside Ava; existing models are opponents only | User decision |
| End goal only stated | Rules 1 and 2 plus a ledger with separate "weights" and "architecture" columns | Makes the goal checkable |
| D3: Qwen-2.5-0.5B SMF retrofit | Our own child-mind from scratch | Rule 1 |
| D1 source requirement open | Resolved by Design D (records with sources) | Design D chosen |
| D2 write via SMF gradients | Resolved: append a record, no gradient | Design D |
| P0: reproduce SMF to build on it | SMF is an opponent; reproduce it only to claim to beat it (rule 8) | Rule 1 |
| E2 on a memory layer's lookup score | M1 on Design D's signal, same statistics | Design D replaced the memory-layer design |
| No step before experiments | Scoreboard first (SB) | Every entrant needs the same race track |
| Design D's matcher: pretrained cross-encoder | Trained from scratch | Rule 1 |
| S1: CPU/SSD cost of a memory-layer model; S2: SMF vs replay | Dropped as separate steps. Replay at swept capacity is an M2 opponent; CPU cost is test T5. | Design D does not use a memory layer |
| Robot experiments R1–R4 | Moved to "Later organs"; the designs remain in survey §12.7 | Stage 1 now starts from our own child-mind |
| C1 kill: worse than a same-size Transformer | Gates G1 (overfit) and G2 (reading EM ≥ 0.95, unknown ≥ 0.95) | C1 is itself a small Transformer, so the old test compared it with itself |
| M1 confirmatory population "(ii)+(iii)" | (i)+(ii)+(iii) | (ii) and (iii) hold no correct answers, so AUROC was undefined |

### v1 → v2 (2026-09-24)

| v1 | v2 | Reason |
|---|---|---|
| "The one part nobody has built" | Narrowed to the calibration question | SMF, Larimar, GRACE, LMLM [C✓] |
| "Learning by writing to memory" was new | Prior work | SMF: 11% vs 89% forgetting [C✓] |
| Answers "with its source" | Open decision D1 | Product-key slots carry no provenance |
| Writes run on a laptop, no GPU | Unknown; to be measured | No write mechanism was specified |
| 20M from scratch as the main test | Public 0.5B retrofit | 20M tests template memorisation; 13× parameter confound |
| E0 threshold: 2,000 lookups/s | p99 ≤ 100 ms, table ≥ 2× RAM, O_DIRECT | The v1 threshold was already passed; a 512 MB table sat in page cache |
| E2 labels by exposure; weak baselines | Labels by correctness; strong baselines; name-controlled strata | Trivial novelty detection could win |
| "Significantly better", 3 seeds | One contrast, margin 0.03, cluster bootstrap, 5 seeds | v1 statistics were undefined |
| "A 1T model's knowledge fits on disk" | Deleted | Confused bits of information with bytes of storage |
| "27 of 27 quotes checked; every claim links" | 38 key quotes re-checked; other [C] claims read once | v1 overstated its own verification |
| R1 forgetting vs 2% replay | Storage and compute axis | 2% replay already reaches near-zero forgetting |
| Month 1: E0–E2 | Months 1–4 | Unrealistic for one person |
