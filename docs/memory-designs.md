# Memory design candidates (round 1)

Date: 2026-09-25. Every design here is **[H]**: a hypothesis, not tested. Evidence references ("§") point to `foundational-survey.md`.

## What the memory must do

Requirements, from the failure table of existing memories (`plan.md` v2 §3, survey §13):

| # | Requirement | Existing designs that fail it |
|---|---|---|
| R1 | Learn a fact instantly, with no gradients (cheap on CPU) | Memory layers / SMF (needs gradients), fine-tuning |
| R2 | Do not forget old facts | Fine-tuning; SMF forgets less but learns less |
| R3 | Answer reworded questions | GRACE "fails to generalize" to paraphrases |
| R4 | Know the source of each fact | Memory layers (slots are shared numbers) |
| R5 | Calibrated "I don't know", catching near-misses | All. RAG near-misses, retrieval lowers abstention, similarity thresholds near chance (§4.5), "memory traps" on robots (§13) |

The test for every design: the cat/dog example. The memory holds "Priya's cat is Mochi"; the question is "What is Priya's dog called?". A design passes only if it says it does not know.

---

## Design A: Factored memory ("who" and "what" matched separately)

**Idea.** Store each fact as a record: *(subject, relation, value, source, time)*. A small trained encoder turns any question, however worded, into a subject query and a relation query. Lookup matches **subject and relation separately** and reports two scores, not one.

**Why it might catch near-misses.** Existing retrieval collapses everything into one similarity number. "Priya's dog" vs "Priya's cat" scores high because most words overlap. Factored matching exposes it: subject match high, relation match low, which means "I know about Priya, but not this". That is a *partial-knowledge* answer, which is more useful than a flat "I don't know".

**Meets R1** (a write is appending a record), **R2** (records do not interfere), **R4** (the source is stored in the record). **R3** depends on the question encoder. **R5** is the core claim.

**Known risks**
- Knowledge must be triple-shaped. Stories, skills and fuzzy knowledge do not fit.
- The question encoder can map the relation wrongly. GrailQAbility documents exactly this: a knowledge-base QA system substituted a missing relation and returned 152 wrong answers (§4.5). This design bets that reporting the relation score *separately* exposes that substitution, while prior systems hid it.
- Nearest prior work: LMLM (triplet database, "return unknown" below similarity 0.6), knowledge-base QA. The difference is two calibrated scores plus partial-knowledge answers. The red team must check whether that difference is real.

**Cheapest kill test.** Build synthetic facts with near-miss probes (same subject, different relation; same relation, different subject). If the factored scores do not separate near-misses better than a single-vector similarity at equal encoder size, Design A is dead.

---

## Design B: Pattern-separated episodic memory with a check step (hippocampus-inspired)

**Idea.** Store raw episodes (text + source + time), like the hippocampus. Two brain-inspired mechanisms:
1. **Pattern separation for addressing.** Project each episode into a very high-dimensional, very sparse code (only ~1% active, like the dentate gyrus: "0.5–1% population sparseness", §3). Similar-but-different memories then get clearly different addresses, which directly targets near-misses.
2. **A check step before answering.** After retrieving, the core model asks: "Does this memory actually answer *this* question?" If not, it abstains. This is a verification pass, not a similarity threshold.

**Meets R1, R2, R4** by construction. **R3** comes from the retrieval encoder. **R5** comes from the check step plus separated codes.

**Known risks**
- The check step costs an extra model pass per answer. That matters on a CPU.
- Google's "Sufficient Context" work already uses a check of whether the context is enough (§4.5), and models still answered wrongly when it was not. The check is only as good as the core model.
- Sparse random codes are old (Kanerva SDM, fly-inspired hashing [R]). Novelty would have to be in the combination and in the measured result.

**Cheapest kill test.** Same near-miss probes. Compare three versions: (a) plain dense retrieval, (b) pattern-separated retrieval, (c) with the check step. If (b) does not beat (a) on near-miss rejection, pattern separation is dead. If (c) adds nothing over (b), drop the check step.

---

## Design C: Two memories that consolidate, like sleep (complementary learning systems)

**Idea.** Human memory has a fast store (hippocampus) and a slow one (cortex). New memories are consolidated from fast to slow during sleep (McClelland 1995, Kumaran 2016, §3). Copy that:
- **Fast store:** Design A or B. Instant writes, sources, no forgetting.
- **Slow store:** a memory layer, updated with sparse memory finetuning during idle time (a robot charging at night). It generalises better to rewording.
- **Confidence from agreement.** If fast and slow agree, answer confidently. If they disagree, or only one knows, answer carefully and say why.

**Meets R1** (fast store), **R2** (sparse slow updates + replay from the fast store), **R3** (the slow store generalises), **R4** (the fast store keeps sources), **R5** (agreement signal).

**Known risks**
- The most complex design, with the most to go wrong.
- Dual-memory continual-learning methods exist [R, needs a search]. Novelty would be in the agreement-based calibration.
- It needs A or B to work first.

**Cheapest kill test.** Only after A or B passes. Does the agreement signal beat either store's own confidence on E2's near-miss strata?

---

## Proposed order

1. **Red team all three** for prior art and flaws, before any code.
2. **A and B share the same kill test**: near-miss probes on synthetic facts, CPU-only, tiny encoders. Build that test first. It is also the start of plan v2's E2.
3. **C** only if A or B survives.

## Open questions for the red team
- Has factored subject/relation matching with separate calibrated scores been evaluated for abstention? (A)
- Has pattern-separated (very sparse, high-dimensional) addressing been tested for rejecting near-misses in retrieval? (B)
- Has fast/slow agreement been used as an uncertainty signal in continual learning? (C)
- Which design fails R3 worst on real paraphrases?

---

## Red-team verdict (2026-09-25)

An adversarial review read the prior work and checked the designs. I verified its 10 key quotes word for word in the downloaded papers (**[C✓]**). Its FlyHash check is one seed on synthetic random vectors, not real encoder embeddings.

### Something this file got wrong
GrailQAbility was cited only as a failure. On the same benchmark, **RetinaQA** (https://arxiv.org/abs/2403.10849) handles the cat/dog case well once trained with unanswerable questions and a tuned threshold: F1 94.30 when a fact is missing, 88.52 when a relation is missing [C✓]. Its reason: "discriminative models are generally better calibrated" [C✓].

Near-misses remain unsolved for **open-text** memory. They are largely solved for **fixed-schema** knowledge bases.

### Design A (factored subject/relation): partly done; narrow gap open
- **KBLaM** (ICLR 2025) stores (name, property, value) triples and learns "to refuse to provide an answer if the information required to answer the question is not present in the KB" [C✓]. But its unanswerable questions are "irrelevant to any <name> or <property> in the KB" [C✓]. So the known-subject, missing-relation case (cat/dog) was never tested.
- **Facts as Experts** (Google 2020): memory keyed by (subject, relation), with a learned "null" answer [C]. Its single score is plausibly a sum of A's two scores [H].
- **LMLM**: separate entity and relation slots, but a single fused cosine with a 0.6 threshold; abstention not evaluated [C✓].
- Calibration is known to be harder when knowledge is incomplete ("much less effective under the OWA") [C✓].
- **Flaws:**
  - The cat/dog difficulty moves into the relation encoder: "cat name" vs "dog name" are close.
  - Two-hop facts (Priya → pet → Mochi → species → cat) defeat a one-triple match.
  - Open relation vocabularies turn relation matching back into fuzzy text matching.
- **Kill-test fix:**
  - Held-out relations with paraphrased wording.
  - Near-misses stratified by closeness.
  - Equal-capacity baselines: a single-vector encoder with the same hard negatives, a single cosine + gap into the same combiner, an NLI cross-encoder verifier, LMLM's threshold.
  - A real-data smoke check on LongMemEval's abstention questions, built exactly the cat/dog way (memory: "10-gallon tank, which has my betta fish"; question: "How many fish are there in my 30-gallon tank?") [C✓]. Only 30 such questions, too few for significance.

### Design B (pattern separation + check): addressing half dead; check step already done
- Random sparse expansion cannot improve discrimination: similarity on the sparse codes is a "monotonic function of the angles or cosine similarity between the input vectors" [C✓], so AUROC cannot rise. The red team's toy check agreed: FlyHash never beat the cosine it was computed from (0.711 vs 0.745 at the hardest setting) [one seed, synthetic].
- In the brain, pattern separation reduces interference when *storing*. It does not help reject a *lookup*. This file conflated the two.
- The check step exists already: CRAG's retrieval evaluator [C], Sufficient Context, and GRAB-RAG, where small models "abstain reliably when evidence is missing" but still answer 41.6% of questions with misleading passages [C✓].

### Design C (fast/slow agreement): mostly done; the agreement signal failed
- Larimar already frames itself this way: "We follow the CLS view, where a hippocampal fast-learning system records samples as episodic memory" [C✓].
- GRAB-RAG tested agreement as an abstention gate (answer only if the context answer matches the closed-book answer). It cut misleading-context answers to 13.3%, but correct answers discarded rose from 0.0% to 49.8% [C✓].
- The two stores are correlated by construction. Both can agree on "Mochi" for the dog question.

### New candidate: Design D, per-subject discrimination with "none of these"
Suggested by the prior work (RetinaQA's discriminator + KBLaM's name/property keys):
1. Link the subject ("Priya").
2. List only that subject's stored facts, plus an explicit **"none of these"** option.
3. A small cross-encoder, trained on near-miss negatives, scores the question against each option.

This turns open similarity into a closed choice among a few options, which RetinaQA argues calibrates better.

**Rule 1 (plan v3):** the matcher is trained from scratch by us. Pretrained cross-encoders (MiniLM, NLI models) appear only as opponents on the scoreboard. The partial-knowledge answer comes for free: "I know Priya: cat, job, city. Nothing about a dog." Writes stay instant (append a record, with its source). It runs on CPU with a handful of small cross-encoder passes per question [H, not timed].

**Where the invention is, honestly.** The parts exist: subject linking, discriminative ranking, a "none" option. What is new, as far as about 12 searches found:
- **Open-text** relations, with no fixed schema.
- **Calibrated** known-subject/missing-relation abstention, measured against strong baselines.
- Learning new facts and new *kinds* of facts on the fly while keeping that calibration.
- All on a laptop CPU.

That is a narrow but real target, and it is the memory Ava needs.

**Kill test for D (to finalise before any run)**
- Synthetic facts with open-text relation phrasings.
- Held-out relations and paraphrases; near-misses stratified by closeness.
- Baselines: single cosine + threshold (LMLM style), fused-key refusal (KBLaM style), NLI verifier, single-vector encoder with the same hard negatives.
- Primary: AUROC for "answerable vs not" on known-subject/missing-relation probes.
- Kill if D does not beat the best baseline (chosen on dev) by a pre-set margin with a CI excluding zero.
- Smoke check on LongMemEval's abstention set.

### Ranking now
1. **D.** Cheapest (CPU), the most specific gap, directly usable in Ava 0.
2. **A.** Folded into D as an ablation (factored vs single score).
3. **B**'s check step becomes a baseline for D. Its pattern-separation half is dropped.
4. **C** is shelved. Its agreement signal failed in published tests.
