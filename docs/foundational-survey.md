# Foundational survey: a brain-like AI that runs on a low-end laptop

Date: 2026-09-24. Compiled from primary sources: papers, code, official model cards and datasheets. Sections 1 to 11 cover the general survey; section 12 covers the robot-brain stages.

## How to read the tags

| Tag | Meaning |
|---|---|
| **[V]** | Measured or read directly for this document. The command or quote is given. |
| **[C]** | A primary source says it. The sentence is quoted and the link given. |
| **[S]** | Seen only in a search snippet, press article, vendor page or summarising fetch. Weaker. |
| **[R]** | From memory, not checked against a source. Treat as unverified. |
| **[H]** | Hypothesis or my own arithmetic. Plausible, untested. |

27 headline quotes (15 in sections 3 to 5, 12 in section 12) were re-checked word for word against the downloaded paper text; all appeared verbatim and are marked **[C✓]**. The other [C] claims were read once and not re-checked.

Papers are not stored in the repository; re-download them from the links.

---

## 1. The goal and the decision

**Goal:** an AI that learns and runs like a brain. It doesn't need terabytes of storage or heavy compute. It runs on a low-end laptop, CPU is enough, GPU/TPU are optional. If it knows something it answers; if not, it says so and offers to search or use a tool. Eventually better than current LLMs.

**Decision this survey informs:** which architectural direction to prototype first, and which capabilities milestone 1 targets.

**Threshold set before reading.** A direction qualifies for milestone 1 only if it has all three:
- (a) a published language-modelling (or close) result against a same-size baseline at comparable compute;
- (b) inference and learning plausibly feasible on the laptop below;
- (c) a named failure cause that a new design could attack.

**Current setting:** none. No project exists yet.

### North star (set 2026-09-24)

The invention is meant to be **the brain of the final form of AI**: the general AI, robot or "superhuman" system people are converging on, whatever it ends up being called. Every milestone is judged by whether it moves toward that brain.

Why the robot-brain goal strengthens the design principles [H, reasoning; hard robot constraints are researched in section 12]:
- **Low power.** A robot carries its compute and battery. A brain that needs a GPU cluster cannot live inside it.
- **Learning on the job.** A deployed robot meets new people, objects and places daily. This is the long-term memory gap on which GPT-4 and GPT-5 both score 0/10 (section 3).
- **Knowing what it doesn't know.** For a robot, bluffing is a safety failure, not just a wrong answer.

### Roadmap

Each stage must produce a measured result on its own before the next starts. **Superseded in detail by `plan.md` v3**, which also sets Rule 1: no borrowed trained model inside Ava.

| Stage | Adds | Capability groups (section 3) | First experiments |
|---|---|---|---|
| 1. Memory and honesty | Our child-mind + honest memory (Design D) + "do I know this" signal | B, C, parts of D | Scoreboard, C1, M1, M2 (`plan.md` v3 §7) |
| 2. World model | Predicting consequences of actions, cause and effect, planning | A (world model, planning) | Candidates in section 12.7, to be re-planned |
| 3. Senses and body | Perception and motor control, in a free simulator first | E, F | Ava organs I2 to I5 (`ava-blueprint.md`); robot candidates in section 12.7 |

---

## 2. The hardware, measured

| Quantity | Value | How | Tag |
|---|---|---|---|
| CPU | Intel i5-11300H, 4 cores / 8 threads | `lscpu` | [V] |
| RAM | 15.4 GiB visible, ~11 GiB free at idle | `/proc/meminfo`, `free -m` | [V] |
| RAM spec | DDR4-3200, 2 channels, AVX-512 | Intel ARK page | [C] |
| RAM theoretical peak | 51.2 GB/s (3200 MT/s × 8 B × 2 ch), only if both channels are populated | arithmetic | [H] |
| RAM measured read | 18.1 to 19.8 GB/s best, 4 to 8 workers | `membw.py` (Appendix A), 2 runs | [V] **provisional** |
| SSD | KIOXIA KBG40ZNV512G NVMe, 477 GB, 199 GB free | `lsblk`, `df -hT` | [V] |
| SSD sequential read | 2.1 GB/s | `dd if=… bs=64M iflag=direct`, 4 GiB | [V] **provisional** |
| SSD random 4 KiB read | 9.2k to 11.6k IOPS at 1 thread (p50 76 to 113 µs); 38k to 68k IOPS at 16 to 32 threads | `ssdrand.py` (Appendix A), 2 runs | [V] **provisional** |

**Why the measured rows are provisional.** Another training job was using about 6.6 of 8 CPU threads during every measurement, and a browser was open. The RAM figure also comes from `numpy.sum`, so it is a lower bound. The gap between 19.8 measured and 51.2 theoretical is unexplained: it could be load, a single populated channel, or numpy overhead. A clean re-run, at low priority and light load, is pending.

---

## 3. What capabilities the AI needs

The checklist below merges 13 frameworks:
- CHC theory (McGrew 2009)
- Hendrycks et al. 2025, "A Definition of AGI"
- Morris et al., "Levels of AGI"
- Legg & Hutter 2007
- Chollet 2019
- Lake et al. 2017
- Botvinick et al. 2017
- LeCun 2022
- Soar (Laird 2022)
- ACT-R (Anderson 2004)
- Kotseruba & Tsotsos
- Complementary learning systems (McClelland 1995, Kumaran 2016)
- CoALA (Sumers 2023)

Sources are listed in Appendix B.

| Group | Capabilities | Current LLM status |
|---|---|---|
| A. Learning and reasoning | Novel-problem reasoning; few-shot skill acquisition; compositionality; causal world model; planning that compiles into fast skill; core priors (objects, agents, number, geometry); theory of mind | Fluid reasoning is weak. ARC-AGI-2 (May 2025 table): o3 (Medium) 3.0%, another frontier model 0.9%; every task was solved by at least two human testers [C]. Leaderboard not re-checked since. |
| B. Memory | Working memory; long-term storage without forgetting; episodic; semantic; procedural; precise retrieval with a failure threshold; consolidation and replay | **Long-term memory storage: GPT-4 and GPT-5 both 0/10** (Hendrycks et al.). "Long-term memory storage is perhaps the most significant bottleneck, scoring near 0% for current models." [C✓] |
| C. Metacognition | Confidence that tracks correctness; knowing when to ask for help or use a tool; a self-model | Partial. See section 4.5. |
| D. Knowledge | Common sense; reading and writing; maths; domain expertise | The strongest area for LLMs. GPT-5: K 9/10, RW 10/10, M 10/10 [C] |
| E. Perception, speed, action | Vision; audio; real-time operation; motor (some frameworks exclude this) | GPT-5: V 4/10, A 6/10, S 3/10 [C] |
| F. Control | Attention and executive control; intrinsic drives; autonomy | LeCun calls the configurator "the most mysterious" module [C] |
| G. Efficiency | Low compute and energy; sparse activity | Only Chollet and the neuroscience give efficiency a formal place. Legg & Hutter call resource limits "superfluous, or wrong" [C] |

**Findings from this strand**
- **"Knowing what you know" in humans is inferred from retrieval, not read from an internal monitor.** Koriat 1993: feeling-of-knowing "is parasitic on the processes involved in attempting to retrieve the target" [C, abstract]. ACT-R turns this into a mechanism: a memory chunk is "retrieved only if [its] activation is over a threshold" [C]. That is a tested cognitive model of the "I don't know" behaviour Ava needs. It carries retrieval's biases.
- **Averaging capability scores hides fatal gaps.** Hendrycks et al. say so themselves: "An AI system with a 90% AGI Score but 0% on Long-Term Memory Storage (MS) would be functionally impaired by a form of 'amnesia'" [C]. Fourati 2025 proposes a coherence-based score that puts GPT-5 at 24% rather than 57% [S].
- **Why classic cognitive architectures faded.** Soar "cannot process unrestricted language" and has neither core knowledge nor large commonsense knowledge [C]. CoALA: they "require many pre-specified rules to function" [C]. They failed on language and on learning at scale, which is exactly where LLMs are strong.
- **Brain numbers, now sourced:**
  - 86.1 ± 8.1 billion neurons (Azevedo 2009) [C]
  - ~0.15 × 10^15 synapses in the neocortex alone (Pakkenberg 2003) [C]
  - fewer than ~1% of neurons substantially active at once (Lennie 2003) [C]
  - of the "20 W", ~0.1 W is cortical computation and ~3.5 W is long-distance communication (Levy & Calvert 2021) [C]
  - The brain's cost is dominated by moving signals, the same bottleneck as the laptop (section 2).

---

## 4. Census by strand

### 4.1 How much a model can know per parameter

| Source | Result | Conditions | Tag |
|---|---|---|---|
| Allen-Zhu & Li 2024, Physics of LMs 3.3 (arXiv 2404.05405) | "GPT2 … consistently achieves a 2bit/param capacity ratio across all data settings after sufficient training" | Synthetic (name, attribute, value) facts; 1000 exposures per fact; ≤0.5B params [S for the size range] | [C✓] |
| same | 1 bit/param at 100 exposures; "quantizing to int4 reduces capacity to 0.7bit/param"; int8 lossless | Post-training GPTQ | [C✓] |
| same | Junk data at a 1:7 useful-to-junk ratio cuts capacity up to 20x at 100 exposures | Synthetic | [C] |
| Morris et al. 2025 (arXiv 2505.24832) | "approximate capacity of 3.6 bits-per-parameter" | Uniform random tokens; models 100K to 20M params | [C✓] |
| Hoffmann et al. 2022, Chinchilla | Training compute ≈ 6ND; ~20 tokens per parameter (from Table 3, not stated in words) | | [C] |
| Besiroglu et al. 2024 | Chinchilla Approach 3 fit is flawed; refit gives 25.6 tokens/param | | [C] |
| Kumar et al. 2024 | Post-training quantization damage grows with training data; 7 to 8 bits compute-optimal for training | ≤1.7B params, loss only | [C] |
| Dettmers & Zettlemoyer 2022 | 4-bit post-training quantization near-optimal; 3-bit reverses it | 19M to 176B, pre-2023 models | [C] |
| SparseGPT / Wanda | 50 to 60% unstructured pruning at little cost on large models | OPT-175B trained at ~1 token/param, so heavily undertrained | [C] |
| Xiao et al. 2024, Densing Law | Capability density doubles every ~3.3 months; pruning and distillation "usually cannot" raise density | Benchmark-based; contamination caveat stated | [C] |
| Mallen et al. 2023 | On the least popular PopQA questions: "GPT-j 6B has 16% accuracy and GPT-3 davinci-003 has 19%"; a retrieval-augmented GPT-neo 2.7B beats davinci-003 there | | [C✓] |
| Kandpal et al. 2023 | Fact accuracy tracks how many training documents mention the fact; the paper contradicts itself on the extrapolation (10^15 vs 10^18 params) | | [C] |
| Bartol et al. 2015 | ~4.7 bits per synapse (26 distinguishable strengths) | Rat hippocampus, 287 synapses, spine-size precision rather than demonstrated storage | [C] |
| Landauer 1986 | Human functional memory "around 10^9 bits" | Abstract only; methods not read | [S] |

**Open question no source answered:** how many parameters *skills* (language, reasoning) need, separately from knowledge.

### 4.2 Associative and sparse-lookup memory

| Approach | Best language result vs baseline | Cost / where memory lives | Failure or limit | Tag |
|---|---|---|---|---|
| Kanerva SDM | None on language | Address search compares against all locations (dense) | Joint training with an encoder: "many dead neurons and failure to continually learn" (Bricken 2023) | [C] |
| Modern Hopfield (Ramsauer 2020) | None on language | O(N·d) softmax over all stored patterns, the same cost as attention | Exponential capacity proven only for random patterns on a sphere | [C] |
| Hyperdimensional computing / VSA | No generative LM found in two searches | Cheap bind/bundle | Kleyko survey: word-vector uses "largely overshadowed" by word2vec and GloVe | [C] |
| Product Key Memory (Lample 2019) | 12 layers + 1 memory: ppl 15.6 vs 24 layers: 16.0 | O((√K + k²)d) lookup; 1M slots | Without query BatchNorm, only 25.8% of slots were used | [C] |
| Memory Layers at Scale (Meta 2024) | 1.3B base, NQ: dense 7.76, Memory+ 64M keys 20.78, Llama2-7B 25.10 | Up to 128B memory params on GPU with custom kernels | "sparse and dense layers are both needed"; replacing more FFN layers degrades quality; unstable for small bases | [C✓] |
| PEER (DeepMind 2024) | C4 ppl 16.45 vs dense 18.31 at 2e19 FLOPs | ~1M single-neuron experts | No public code | [C] |
| UltraMem / V2 (ByteDance) | 1.6B-x12 loss 2.24 vs Dense-6.5B 2.30 | A100 only | Lost to MoE at 151M; V2 "underperforms" on maths, code, reasoning | [C] |
| Engram (DeepSeek, Jan 2026) | MMLU 60.4 vs MoE 57.4 at equal activated params | O(1) hashed n-gram lookup; 100B-param table in host DRAM cost ≤2.8% throughput (GPU server) | "memory cannot replace computation in this regime"; optimum is ~20 to 25% of sparse params in memory | [C✓] |
| kNN-LM (2020) | WikiText-103 ppl 18.65 → 16.12 | 293 GB fp16 datastore | Gains from n-gram overlap (Drozdov); "does not improve open-ended text generation" (Wang 2023); no gain on rare tokens (Nishida 2025) | [C] |
| RETRO (2021) | ≈GPT-3 on the Pile with 25x fewer params | 2T-token database, 10 ms lookup | Leakage admitted by the authors; Norlund 2023: gains "largely originate from overlapping tokens"; NVIDIA reproduction: real gains on knowledge tasks only | [C] |
| Atlas (2022) | NQ 42.4% 64-shot, beating PaLM 540B by 3 points | Index 49 GB, 4 GB after PQ compression | Stale index; 11B reader | [C] |
| LLM in a flash (Apple 2023) | Same quality as the source model | M1 Max CPU: 0.67 to 1.0 s/token for 7B | Needs ReLU sparsity; used "best theoretical" IO latency | [C] |
| PowerInfer (2023) | Same quality; 13.2 tok/s quantized, up to 11.69x llama.cpp | i9 + RTX 4090 | SwiGLU models only 1.5 to 1.7x faster | [C] |

**No memory-layer language model has ever been benchmarked on CPU-only inference.** Every CPU-feasibility claim for this family is arithmetic [H].

### 4.3 Brain-inspired learning and hardware

| Approach | Best language result vs baseline | Learning rule actually used | Tag |
|---|---|---|---|
| HTM (Cui 2016) | Reber grammar 98.4% vs LSTM 100%; no LM benchmark | Local, one-pass | [C] |
| HTM benchmark critique | Wu & Keogh: NAB and other anomaly benchmarks "irretrievably flawed" | | [C] |
| Monty / Thousand Brains | None (3D objects only); ~528,000,000x fewer training FLOPs than a ViT | Local associative | [C] |
| SpikeGPT 216M | WikiText-103 ppl 39.75, worse than zero-shot GPT-2 Small at 37.50 (SpikeGPT's own table swaps the GPT-2 columns) | Backprop with surrogate gradient | [C] |
| SpikeBERT / SpikeLM | GLUE 80.2 vs BERT 84.3; 76.5 vs 83.2 | Backprop + distillation | [C] |
| SpikingBrain-7B | MMLU 65.84 vs Qwen2.5-7B 74.21, and that score is the floating-point model | Converted from a Transformer | [C] |
| Spiking-LM energy savings | Every figure estimated from 45 nm per-operation tables; none measured on silicon | | [C] |
| IBM NorthPole | 3B model at 28,356 tok/s on 16 cards, 672 W; HumanEval pass@10 0.292 vs 0.300 on GPU | Backprop + quantization-aware training; not spiking | [C] |
| Forward-Forward (Hinton 2022) | Next-character result retracted: "I now suspect it was due to a bug" | Local | [C✓] |
| Direct Feedback Alignment (Launay 2020) | WikiText-103 ppl 52.0 (macro) / 93.3 (micro) vs backprop 29.8; still partly backprop | Mostly local | [C✓] |
| Split forward gradient (2026) | 16M model, ppl 387 vs backprop 150 | Forward gradient | [C] |
| Predictive coding | Matches backprop to ~7 layers, falls behind deeper (ResNet-18 CIFAR-10: 70.44% vs 92.83%); ~100x backprop compute | Local | [C] |
| Equilibrium propagation (Kerjan 2026) | ImageNet VGG10 top-5 error 13.23% vs backprop 12.2%. Largest backprop-matching local-rule result found. Vision only. | Local, simulated on GPU | [C] |
| EWC continual learning | Split-MNIST class-incremental: 20.01%, the same as no protection (19.90%); replay 90.79% | | [C✓] |
| Continual pre-training (Ibrahim 2024) | LR re-warm + re-decay + 5% replay matches full retraining at 405M and 10B | | [C] |
| Organoids (DishBrain, Brainoware, FinalSpark) | No language capability. FinalSpark's only demonstrated compute: detecting a 20 Hz stimulus at 95.5%. Organoids live up to 100 days. | Biological plasticity | [C] |
| DishBrain critique | Balci et al. (30+ neuroscientists): "strong claims … with relatively weak evidence" | | [C] |
| Organoid access cost | FinalSpark ~$500/user/month (2024 press); Cortical Labs CL1 $35,000 or $300/week (2025 press) | | [S] |

### 4.4 Efficient architectures and CPU speed

| Architecture | Result vs same-size Transformer | CPU measurement | Tag |
|---|---|---|---|
| BitNet b1.58 700M / 3B | Average −1.2 / +0.5 vs a self-trained LLaMA (100B tokens, same data) | None in paper (GPU latency) | [C] |
| BitNet 2B4T | MMLU 53.2 vs Qwen2.5-1.5B 60.3 (different data: 4T vs 18T tokens) | 29 ms/token on an i7-13800H, 8 threads; 0.4 GB non-embedding memory | [C] |
| bitnet.cpp "100B on CPU" | n/a | 1.70 tok/s on the i7, but using **untrained dummy models**; baseline was fp16 llama.cpp, not 4-bit | [C] |
| TriLM (Spectra), same data | 99M to 560M: 0.8 to 4.3 points below float; "TriLM 3.9B matches the performance of FloatLM 3.9B" | | [C✓] |
| BitNet Reloaded (≤48M) | Ternary needs ~2x hidden width to match 16-bit | | [C] |
| MatMul-free LM | 370M: 40.3 vs Transformer++ 41.1; the 13B efficiency claim used random weights | GPU and Loihi 2 only | [C] |
| Mamba-2 350M, same data | ppl 8.60 vs Transformer++ 8.68; **with ~10% attention layers 8.26** (best) | None | [C] |
| NVIDIA 8B Mamba study | Pure Mamba fails phonebook recall "beyond approximately 500 tokens", even after 3.5T tokens; a hybrid with 7% attention beats the Transformer on all 12 tasks | None | [C✓] |
| Zoology (70M to 1.4B) | "82% of the gap is explained by … associative recall" | None | [C] |
| RWKV-7, xLSTM | Beat Pythia or self-trained Llama baselines at 125M to 420M | None | [C] |
| TinyStories | Models under 10M params write fluent children's stories | llama2.c 15M: ~110 tok/s on an M1 MacBook Air | [C] / [S] |
| llama.cpp | Token generation is memory-bandwidth bound: Llama-2-7B on M1 Pro gives F16 12.75, Q8 22.34, Q4 36.41 tok/s | Metal GPU, same mechanism | [S] |

### 4.5 Hallucination and "knowing what you know"

**Causes**
- **Rare facts.** Kalai & Vempala 2023: "the probability of generating a hallucination is close to the fraction of facts that occur exactly once in the training data" [C]. Their argument covers only arbitrary facts that appear once. They say systematic facts may be fixed by "different architectures".
- **Scoring rules reward guessing.** Kalai et al. 2025 (OpenAI) show most major benchmarks give "I don't know" zero credit, so guessing is optimal. This "holds for arbitrary language models, including those with RAG" [C].
- **Fine-tuning on unfamiliar facts.** Gekhman et al. 2024: facts the model did not already know are learned slowly, and once learned they "linearly increase the model's tendency to hallucinate" [C].

**Support for the explicit-memory idea**
- Kalai et al. 2025: "a non-hallucinating model could be easily created, using a question-answer database and a calculator, which answers a fixed set of questions [...] and otherwise outputs IDK" [C✓]. They add that such a system gives up coverage.
- Self-RAG on PopQA: 24.7 without retrieval, 45.5 with it [C].

**Evidence against "a lookup miss means I don't know"**
- **Near-miss lookups.** GrailQAbility: with a relation deleted from the knowledge base, the model substituted a different relation "and retrieves 152 answers" instead of returning none [C✓]. In Stanford's legal RAG study, 23 to 38% of hallucinations cited a real but inapplicable document [C]. Retrieval-score thresholds give AUROC 0.513 to 0.583 on reasoning-heavy queries, near chance (Magnitude Mirage; unreviewed preprint, Sept 2026) [C].
- **Retrieval reduces abstention.** One frontier model abstained on 84.1% of questions without retrieval and on 52% with it (Joren et al., Google) [C✓]. Gemini 1.5 Pro dropped from 100% to 18.6% [C].
- **Generators ignore good retrievals.** RAG models reject pure-noise context at most 45% of the time (RGB) [C]. Context in the middle of a long prompt can be worse than no context (Lost in the Middle) [C].

**Every abstention method trades coverage for precision**
| Method | Result | Tag |
|---|---|---|
| IDK relabeling | Accuracy on answered questions 43.0% → 61.8%, answering 58.7% | [C] |
| R-Tuning | Answers only 2.41 to 9.56% of MMLU | [C] |
| Semantic entropy (Nature 2024) | Detection AUROC 0.790 vs 0.69 baselines; ~10x generation cost; blind to consistent errors | [C] |
| Hidden-state "truth" probes | 71 to 83% (Azaria & Mitchell), but below chance on negated statements (Levinstein & Herrmann) | [C] |
| Knowing *its own* knowledge | Kadavath et al.: a classifier from model B predicts model A's correctness as well as one from model A (0.8631 vs 0.8633 AUROC), so no evidence of self-specific knowledge | [C] |

**Gap:** no published language model with exact-key explicit memory reports abstention precision and recall on open questions. This rests on the searches done, not an exhaustive one [C, absence].

---

## 5. Matrix

### Unanimous across the sources read
1. **Memory cannot replace compute.** Meta, ByteDance, DeepSeek and the PKM authors all keep a dense core. Engram's optimum puts only ~20 to 25% of sparse params in memory [C✓].
2. **Sparse memory helps knowledge tasks most.** NQ, TriviaQA and long-tail QA gain; maths, code and reasoning gain little or lose (UltraMemV2, Memory Layers 8B MBPP 42.2 vs 44.2) [C].
3. **Pure fixed-state recurrent models fail at in-context recall. About 10% attention fixes it and beats pure Transformers** (70M to 8B, four independent teams) [C].
4. **Batch-1 CPU decoding is bound by memory bandwidth.** Tokens/s scales with bytes read per token [S, llama.cpp discussions #4167 and #3167].
5. **Every competitive spiking or neuromorphic language model was trained with backprop.** Every spiking-LM energy saving is estimated, not measured [C].
6. **Local learning rules fall further behind backprop as depth and task difficulty grow.** No local rule is competitive on language: the best, DFA, gives ppl 52.0 vs 29.8 [C✓].
7. **Retrieval does not eliminate hallucination.** It shifts the failure to near-miss lookups and to generators ignoring evidence [C].
8. **Abstention always costs coverage** [C].

### Contested
- **Ternary vs 4 to 8 bits.** BitNet argues ternary is enough; Kumar et al. argue 7 to 8 bits is compute-optimal for training [C].
- **Whether retrieval gains are real or overlap.** Norlund says overlap; NVIDIA finds real gains on knowledge tasks [C].
- **Whether hidden-state probes read "truth"** [C].
- **HTM vs LSTM on streaming data**, with sources in both directions [C]/[S].
- **Whether predictive coding can be cheaper than backprop.** One paper proves it is lower-bounded by backprop's time complexity; another argues analog hardware could change that [C].

### Correlates with failure
- Unbalanced memory-key usage (no query normalisation).
- Dead units in Top-K / SDM when the input representation drifts.
- Stale memory keys.
- Ternary weights at small width.
- Pure SSMs on few-shot format following.
- EWC in class-incremental settings.
- Deep predictive-coding networks.
- Retrieval on reasoning-heavy or temporal queries.
- Any evaluation that gives "I don't know" no credit.

### Explicitly tried and rejected by their authors
- Flat memory keys (too slow).
- Random or sink keys (inconsistent gains).
- Replacing more than ~3 FFN layers with memory.
- Forward-Forward next-character modelling (retracted).
- Curriculum learning (BabyLM 2023 and 2024).
- More than ~10% attention in hybrids.
- The NAB anomaly benchmark.
- Lamini's "Mixture of Memory Experts" hallucination paper (withdrawn on arXiv 2025-09-03).

### Beliefs I held before the census that the matrix contradicts
- **"Knowledge in addressable memory makes 'I don't know' detectable by construction."** Only for exact keys. Natural-language questions go through a fuzzy mapping step whose measured failure is the confident near-miss. Retrieval context also *reduces* abstention.
- **"Brain-like learning is a viable path to beat LLMs on language now."** No local-rule or spiking model is competitive on language. Every competitive spiking LM used backprop.

---

## 6. Transfer analysis

| Their condition | Ours | Does the claim transfer? |
|---|---|---|
| Memory layers on H100 with custom kernels, 3 TB/s | 4-core CPU, measured ~20 GB/s RAM, 2.1 GB/s SSD | Unknown. Never measured on CPU. [H] Sparse lookups need only KB per token, so plausible. |
| Memory-layer gains at 134M to 8B base, 1T tokens | Trainable: ~20M to 125M, a few B tokens | Partly. UltraMem *lost* to MoE at 151M. Small-scale gains are not guaranteed. |
| Capacity 2 bits/param at 1000 exposures, synthetic facts | Real text, few exposures per fact | Likely lower: 1 bit/param at 100 exposures, up to 20x loss with junk data. |
| Hybrid ~10% attention wins at 350M on 7B tokens | 20M to 125M on Kaggle | Probably. Zoology saw the recall gap from 70M. Kernel support on T4 is unverified. |
| Ternary parity at 3.9B | ≤125M | No. Spectra measures a 0.8 to 4.3 point loss at ≤560M. |
| BitNet 2B4T 29 ms/token on i7-13800H | i5-11300H, 4 cores | Partly. Expect slower. Not measured here. |
| Semantic entropy AUROC 0.79 at 7B to 70B with 10 samples | Tiny model on a CPU | Costs 10x generation. Small-model calibration is unknown. |
| Abstention results on TriviaQA / NQ / PopQA | Our own synthetic fact set | Needs its own measurement. |
| Local rules near backprop on MNIST / CIFAR / ImageNet-VGG10 | Language modelling | No. The best language result is ppl 52.0 vs 29.8. |

---

## 7. Arithmetic on our numbers

All [H]: arithmetic on measured or cited inputs, not a model run.

**1. Dense model speed on this laptop.** Tokens/s ≤ bandwidth ÷ bytes per token.

| Model | Bytes read per token | At 19.8 GB/s (measured, provisional) | At 51.2 GB/s (theoretical) |
|---|---|---|---|
| 125M params, 8-bit | 0.125 GB | ~158 tok/s | ~410 tok/s |
| 1B params, 4-bit | 0.5 GB | ~40 tok/s | ~102 tok/s |
| 7B params, 4-bit | 3.5 GB | ~5.7 tok/s | ~14.6 tok/s |
| 1T params, fp16 | 2 TB (does not fit in RAM) | ~100 s/token | ~39 s/token |
| 1T params from SSD, 4-bit | 500 GB at 2.1 GB/s | ~4 min/token | n/a |

**2. Knowledge ceiling of a 1T model.** At 2 bits/param the *information content* is at most 2e12 bits = 250 GB; at 0.7 bits/param (int4), ~88 GB. That does not fit in 15 GB of RAM, for any architecture. **Correction (red team):** storing the parameters that hold it takes far more: 500 GB at int4, 1 TB at int8, 2 TB at fp16. All exceed the 199 GB free. An earlier version of this line said it fits on disk. That was wrong: it confused bits of information with bytes of storage.

**3. Sparse lookups from SSD.** Suppose each token needs 100 lookups of 4 KiB:
- At 9.2k IOPS (1 thread): ~92 tokens/s of lookup budget.
- At 38k to 68k IOPS (16 to 32 threads): ~380 to 680 tokens/s.

That is feasible. Latency matters too: p50 is 76 to 113 µs per read, and p99 at high thread counts is 2.3 to 6.0 ms. Lookups must be batched or prefetched. Engram's deterministic addresses allow prefetching [C].

**4. Training budget on Kaggle.** 6ND FLOPs. The T4 fp16 peak of 65 TFLOPS is [R], at an assumed 30% utilization, over 2 GPUs and 30 h/week:
- 125M params: ~5.6B tokens/week.
- 20M params: ~35B tokens/week.

Enough for 20M-scale comparisons across several seeds per week. Not enough for the 300B-token runs behind most published tables.

**5. What milestone 1 can afford.** A 20M model at 20 tokens/param needs 400M tokens, 4.8e16 FLOPs: about 0.34 wall-clock hours on both T4s, or ~0.7 T4-hours, at the assumptions above. Three architectures × three seeds × two sizes fits comfortably in one week.

---

## 8. Recommendation

> **Superseded.** The experiments below (E0 to E3) were revised after red-team review. The current version is `plan.md` v2. Prior work that changes the novelty claim is in §13.

### Which directions pass the threshold from section 1

| Direction | (a) LM result vs baseline | (b) Laptop-feasible | (c) Named failure to attack | Verdict |
|---|---|---|---|---|
| Small dense core + sparse key-value memory (PKM / Engram style) | Yes | Unmeasured; arithmetic plausible | Yes: unbalanced keys, needs a dense core, near-miss retrieval | **Pass. Primary direction.** |
| Hybrid recurrent + ~10% attention core | Yes | Yes | Yes: recall failure of pure SSMs | **Pass. Candidate core.** |
| Ternary / low-bit weights | Yes (parity only at 3.9B) | Yes, measured on i7 | Yes: loss at small width | Pass, but defer; it costs quality at our scale |
| Local learning rules (PC, EP, FF, DFA, Hebbian) | No (best: ppl 52 vs 29.8) | Yes | Yes | **Fail (a).** Research strand, not milestone 1. |
| Spiking networks | Only with backprop | CPU gets no event-driven benefit [H] | Yes | Fail (a) for the "brain-like" claim |
| HTM / Monty / SDM / Hopfield / hyperdimensional | No language result | Yes | Partly | Fail (a) |
| Organoid intelligence | No | No (wet lab or ~$500/month) | n/a | Fail (a) and (b) |

### Proposed direction [H]

Match the brain's split. A small dense "skill" core does language and reasoning. A large sparse, addressable knowledge memory is looked up a few entries per token, and can live on SSD. The memory's lookup statistics feed an explicit "do I know this" signal, ACT-R style, which gates between answering, abstaining, and calling a tool.

This is supported by findings 1 to 3 and 7 in section 5. The novel and untested part is the abstention signal. That is where this project's contribution could be, and also where the known failure (near-miss lookups) sits.

### Experiments, cheapest first

Each has a kill criterion stated before it runs.

**E0. CPU lookup benchmark (laptop, 0 GPU-hours).**
- Build a 1M × 256 fp16 product-key memory table, 512 MB, in RAM and on SSD.
- Time the batched lookups per token on the i5, with clean hardware measurements first.
- **Kill:** if 100 lookups per token cannot sustain ≥20 tokens/s from SSD with batching, SSD-resident memory is dropped. Memory must then fit in RAM.

**E1. Does memory help at our scale? (Kaggle, ~5 T4-hours.)**
- Tiny Transformer vs the same model with 1 PKM layer vs a same-FLOPs deeper Transformer.
- ~20M params, TinyStories plus a synthetic fact set (name → attribute) with controlled exposure counts.
- 3 seeds each; report mean and spread.
- **Kill:** if the memory variant does not beat the equal-FLOP dense model on fact recall by more than the seed spread, the memory direction is dead at this scale. Stop and report it.

**E2. Is "not found" a usable "I don't know"? (Kaggle, ~5 T4-hours; reuses E1 models.)**
- Test set: facts the model saw, plus facts it never saw, plus near-miss probes (a real entity with a missing attribute, the GrailQAbility pattern).
- Compare the AUROC of known-vs-unknown for three signals:
  1. the memory lookup score (top score and top-1/top-2 gap);
  2. the dense model's max-softmax probability;
  3. predictive entropy.
- Use DeLong's test for paired AUROCs. Holm-correct across the signal comparisons.
- **Kill:** if the memory signal's AUROC is not significantly above the best dense baseline signal, the "honesty by construction" idea is dead in this form. Report that at full strength.

**E3 (only if E1 and E2 survive).** Swap the core for a hybrid (~10% attention). Then test adding new facts to memory without retraining the core (continual learning), against 5% replay as the baseline to beat.

**Not recommended now:**
- Local learning rules, spiking or organoid work as the main path. The evidence in 4.3 says none is competitive on language.
- Any claim of "trillion-model capability". Section 7.2 rules it out in RAM by counting.

---

## 9. Corrections to earlier estimates

1. **RAM bandwidth.** The first estimate, ~50 GB/s from memory, was wrong. Measured is ~20 GB/s under load (provisional). Every tokens/s estimate made before measuring was 2.5x too optimistic.
2. **"~2 bits per parameter whatever the architecture."** Too strong. The paper measures ≤0.5B-param models on synthetic facts. Gated-MLP models (LLaMA-style) store 1.3x less at 100 exposures, and int4 drops capacity to 0.7 bits/param.
3. **"If knowledge is in addressable memory, 'not found' becomes a real, detectable event."** True only for exact keys. The literature documents confident near-miss lookups and shows retrieval *reduces* abstention. The idea survives only as a hypothesis to test (E2).
4. **"~100 trillion synapses."** The verified figure is ~150 trillion for the neocortex alone (Pakkenberg 2003). The whole-brain count was not verified.
5. **Earlier training-time table.** It assumed ~25 TFLOPS effective per T4 from memory [R]. That assumption is still unverified.
6. **"Only 3 to 10% of MoE parameters active per token."** Stated from memory and not checked.

---

## 10. Unverified items

- The T4 fp16 peak (65 TFLOPS) and compute capability 7.5 [R].
- Whether Mamba, fla/Triton, RWKV-7, xLSTM CUDA and MatMul-free kernels run in fp16 on the T4. The xLSTM sLSTM CUDA kernel requires compute capability ≥8.0 [C]. The T4's 7.5 is [R].
- Clean RAM and SSD numbers on an idle machine.
- Landauer 1986 methods; the Kanerva 1988 book; the ARC-AGI-2 current leaderboard; whole-brain synapse count; the Sokoloff 1960 source for "20 W".
- Organoid pricing (press only); Struye & Latré's "LSTM beats HTM by 30%" (snippet only); the Numenta 123x BERT figure (vendor claim, no accuracy reported).
- Whether any HDC/VSA generative language model or any HTM language benchmark exists. Not found, and the search was not exhaustive.
- Magnitude Mirage (arXiv 2609.15578) is a 10-day-old unreviewed preprint.

## 11. Code and results

Every measured number in the later work comes from code in this repository (`scoreboard/`, `childmind/`, `mind/`), and every result file under `results/` records the exact command, seed and data digest that produced it.

---

## 12. The robot brain: constraints, world models, robot learning

Stages 2 and 3 were researched the same way. Tags as above. Robot product specs come from vendor pages; vendor figures are not independent measurements.

### 12.1 Hard constraints a robot brain must fit

| Constraint | Value | Tag |
|---|---|---|
| Top onboard computer 2025–26 | NVIDIA Jetson Thor T5000: 517 dense FP8 TFLOPS, 128 GB LPDDR5X, **273 GB/s**, 40–130 W (default mode 120 W) | [C] NVIDIA product page and docs |
| Previous generation | Jetson AGX Orin: 204.8 GB/s, 15–60 W. Orin NX 16 GB: 102.4 GB/s, 10–40 W. Orin Nano: 51–102 GB/s, 7–25 W | [C] NVIDIA |
| Comparison | H100 SXM 3.35 TB/s, up to 700 W, ~12x Thor. This laptop: 19.8 GB/s measured (provisional), 51.2 GB/s theoretical | [C] / [V] / [H] |
| Humanoid batteries | Unitree G1 ~421 Wh, ~2 h; 1X NEO 842 Wh, 4 h, runs on Jetson Thor; Figure 03 2.3 kWh, "5 hours of run time at peak performance" | [C] vendor pages |
| Average whole-robot draw | ~210 W (NEO, G1) to ~460 W (Figure 03 at peak); G1 measured "135 (W) at rest" | [H] capacity ÷ runtime; [C✓] Deniz et al. 2026 for 135 W |
| Share of power to compute | **Not published for any humanoid.** A small wheeled robot on a Jetson Xavier: total 30 to 90 W, motors 16.6% to 54.6%; under autonomous navigation, GPU 37.3% vs motors 16.6% | [C✓] Liu, Shi & Shin (arXiv 2511.20467). The figure text also shows a 64.0% value I could not assign. |
| Control-rate tiers | Joint/servo ~1 kHz (Franka FCI, Figure Helix 02 S0); whole-body 500 Hz (Unitree G1 SDK, Mini-Cheetah); visuomotor policy 50–200 Hz; language/semantic layer 5–10 Hz | [C] |
| Offboard cost | Network adds 13 ms (π0 ideal Wi-Fi) to ~21 ms (π real-time chunking, LAN) | [C] |
| Human reference | ~17–20 W glucose; ~0.1 W cortical computation; spinal reflex 20–45 ms; voluntary reaction ≥100 ms | [C✓] Levy & Calvert (PNAS); [C] Kurtzer 2015 |

**What the data shows**
- Every published humanoid stack is layered by speed: a fast reflex layer, a mid-speed policy, and a slow "thinking" layer. Figure's Helix uses a 7B model at 7–9 Hz plus an 80M model at 200 Hz (Helix 02 adds a 10M model at 1 kHz). GR00T N1 runs 10 Hz plus 120 Hz. [C]
- Onboard memory bandwidth, 200–273 GB/s, is ~12x below an H100 and ~4x (204.8 vs 51.2 theoretical) to ~14x (273 vs 19.8 measured) above this laptop.
- Robot compute modules draw 40–130 W, 2 to 6.5 times the whole brain's 20 W and 400 to 1300 times the brain's ~0.1 W of cortical computation [H, arithmetic].

**What it suggests [H]**
- Only the slow (≤10 Hz) layer can be offboarded. A 13–21 ms network hop exceeds a 1–5 ms control period.
- For the language/semantic layer, bandwidth binds, not TOPS. A 7B 8-bit model on Thor tops out near 273 ÷ 7 ≈ 39 tokens/s. That is consistent with Helix's 7–9 Hz.
- A 10–40 W brain (Orin NX class, ~100 GB/s) limits the semantic model to roughly 1–3B params at low bit-width. That matches the stage-1 bet: a small core plus sparse memory.

### 12.2 Robot foundation models (VLAs)

| System | Params | Rate | Where it runs | Key result | Main failure | Tag |
|---|---|---|---|---|---|---|
| RT-2 (2023) | 5B / 55B | 5 / 1–3 Hz | "multi-TPU cloud service" | Unseen tasks 62 vs RT-1 32 | "the robot does not acquire any ability to perform new motions" | [C] |
| OpenVLA (2024) | 7B | ~6 Hz on RTX 4090 | Off-board GPU | +16.5 points vs RT-2-X; LIBERO 76.5 | int8 runs at 1.2 Hz, 58.1% success | [C] |
| π0 (2024) | 3.3B | 50 Hz action chunks | RTX 4090: 73 ms onboard, 86 ms over Wi-Fi | Beats OpenVLA and Octo | Not every task works reliably | [C] |
| π0 / π0.5 on Jetson (Jetson-PI 2026) | 3.3B | "around 0.7 Hz" on Orin; ~2.2 Hz on Thor | Onboard | ~1.4 s per decision on Orin-50W, 458 ms on Thor | Too slow for reactive control | [C✓] |
| GR00T N1 (2025) | 2.2B | 10 / 120 Hz | Data-centre L40, 63.9 ms per chunk | Real humanoid 76.8% vs Diffusion Policy 46.4% | Short-horizon tabletop only | [C] |
| Figure Helix (2025) | 7B + 80M | 7–9 / 200 Hz | "entirely onboard embedded low-power-consumption GPUs" | No published success rates | No paper | [S] vendor blog |
| SmolVLA (2025) | 0.45B | async chunks | Consumer GPU. CPU deployment claimed, not measured. | LIBERO 87.3 vs π0 86.0 | Single robot type, short tasks | [C] |
| Gemini Robotics (2025) | not disclosed | 50 Hz effective | Cloud backbone + on-robot decoder, ~250 ms end-to-end | | Imprecise points and boxes | [C] |

**Benchmark scores are inflated.** Models above 90% on standard LIBERO "collapse to 0.0%" under generalised settings (LIBERO-PRO) and fall "from 95% to below 30% under modest perturbations" (LIBERO-Plus). OpenVLA goes 76.5 → 1.1 under a camera change. Some models' outputs barely change when the instruction is blanked or corrupted. [C]

**Training cost.** OpenVLA 21,500 A100-hours; GR00T N1 ~50,000 H100-hours; SmolVLA ~30,000 GPU-hours. Training a VLA is out of reach on this budget. Fine-tuning a small one may not be. [C]

### 12.3 World models (stage 2)

| System | What it does | Cost / speed | Result | Failure | Tag |
|---|---|---|---|---|---|
| V-JEPA 2-AC (Meta 2025) | 1B video encoder (1M+ hours of video) + 300M action predictor trained on <62 h of robot video; plans in latent space | 16 s per action on RTX 4090, vs "4 minutes" for Cosmos | Zero-shot pick-and-place 80% / 65% vs Octo 15% / 10% (10 trials per cell) | Camera position hand-tuned; needs image sub-goals; errors accumulate | [C✓] |
| Independent check (DUET-DINO 2026) | Official V-JEPA 2-AC checkpoint in simulation, 7-DoF | | Reach 55%, angled reach 2.5%; DINOv3 model 92% | "struggles with fine-grained control" | [C] |
| DINO-WM (2024) | Frozen DINOv2 features + predictor + planning | 53 s per plan on A6000 | PushT 0.90 vs DreamerV3 0.30 (simulation) | Needs offline data with good coverage | [C] |
| DreamerV3 | Learns a world model and acts inside it; 12M–400M params | 1 A100 per agent; Minecraft diamonds in 9 days on 1 GPU vs VPT's 720 GPUs | 150+ tasks, one configuration | No real-robot results in the paper | [C] |
| Dreamer 4 (2025) | 2B-param world model | Trained on 256–1024 TPU-v5p; 21 FPS on H100 | First offline diamonds | Diamonds in only 0.7% of episodes | [C] |
| Genie 2 / 3 | Interactive generated worlds | Genie 3: 24 FPS, 720p | Minutes of consistency | Limited actions; research preview only; no robot use | [S] blog |

Takeaway [H]: world models plan, but slowly (16 to 53 s per action in the two latent planners measured). No world model here runs at robot control rates on robot hardware. **A fast, cheap world model is an open problem**, and it sits directly on the stage-2 path.

### 12.4 Learning on the job

- **No deployed robot updates its own weights online.** π\*0.6 / RECAP improves from real experience, but "our system is not fully autonomous: it relies on human labeling and effort", and it "collects a batch of data, retrains the model, and repeats, rather than running a fully online RL loop". Each round restarts from the pretrained checkpoint to avoid drift. [C✓]
- **Sequential fine-tuning forgets; replay fixes it, in every study read.**
  - LIBERO 2023: EWC "works worse than" plain sequential fine-tuning.
  - Pretrained VLAs without replay: π0 NBT ~0.70 (NBT is negative backward transfer; higher means more forgetting). With a 20% replay buffer: ~0.
  - Replay works "even with a small replay data size (e.g., 2% of training data)".
  - Real-world π0.5 on 10 tasks: naive fine-tuning causes "severe catastrophic forgetting", and replay beats joint training.
  - All of these retrain on data-centre GPUs. [C]
- **Small policies forget most.** Even with replay, small behaviour-cloning policies score NBT 0.13 to 0.25. [C]
- **Brain-inspired option.** Monty learns 77 objects continually on CPU with little forgetting, while a ViT forgets. But the ViT baseline used no replay, and Monty does recognition only, no motor control. [C] / [H]

### 12.5 Robots knowing what they don't know

| Method | Result | Limit | Tag |
|---|---|---|---|
| OpenVLA's own token probability as failure predictor | ROC-AUC **53.83** on unseen tasks, near chance | | [C✓] |
| SAFE (NeurIPS 2025): small detector on VLA features + conformal bound | ROC-AUC ~73 to 85 on unseen tasks | Last-layer features only; cross-robot transfer unknown | [C] |
| KnowNo (CoRL 2023): LLM planner + conformal prediction, asks for help when unsure | Hits the target success rate while "reducing human help by 10−24%"; hardware task success 0.74 vs 0.38 without help | Asks for help often: 0.58 per step. A follow-up measured 77.5% help and 51.3% over-asking at an 85% target. Assumes perception is correct. | [C✓] / [C] |
| Introspective Planning (NeurIPS 2024) | Exact-set rate 93.0%, overstep 3.8% | No statistical guarantee | [C] |

Same pattern as the language side (section 4.5). The model's own confidence is weak, add-on detectors help, and every guarantee costs a lot of help requests.

### 12.6 Brain-inspired and neuromorphic robot control, measured

- **Neuromorphic drone** (Science Robotics 2024): a 28.8k-neuron spiking network running onboard Intel Loihi at 200 Hz, "spending only 27 µJ per inference", in real flight. The task was simple ego-motion control. [C]
- **Neuromorphic force control** (2024): real KUKA peg insertion 100% over 50 runs. Loihi 2 used 53 ± 17 µJ against 3800 µJ on an i7 CPU at similar latency. [C✓]
- Contrast with section 4.3: for **small reactive controllers**, neuromorphic energy savings are measured on hardware. For **language models** they are only estimated. The evidence places brain-like hardware at the reflex layer first.

### 12.7 Stage 2–3 experiments on this budget

All [H]. None has been run. Each needs its kill criterion finalised before it runs.

- **R1. Memory for continual robot learning (Kaggle, LIBERO simulator).** A small policy + sparse addressable memory vs experience replay at equal memory size. Measure NBT and success, and evaluate under LIBERO-PRO or LIBERO-Plus perturbations, not only standard LIBERO. **Kill:** no NBT gain over a 2% replay buffer.
- **R2. Failure detection on an open small VLA.** Reproduce SAFE's finding that token probability is near chance, using SmolVLA (0.45B). Then test whether memory-lookup statistics predict failure better. SmolVLA was run in bf16; the T4 has only fp16, which is untested. **Kill:** memory signal not better than SAFE's MLP.
- **R3. Small latent world model + planner (DINO-WM style) on PushT / Wall / Maze.** Target: the same success at 10x less planning time. **Kill:** no speedup at equal success.
- **R4. KnowNo-style conformal "ask for help" on the stage-1 language core.** Needs about 400 calibration examples, CPU-feasible. Report the help rate as well as coverage.

Not testable here: real-robot latency, neuromorphic hardware (Loihi requires research-community membership), and anything at VLA-training scale.

### 12.8 What changed in my picture

- I said the robot goal strengthens "learning on the job". Confirmed: no deployed system does it online. It is a genuine open problem.
- I suggested fast/slow splits would be needed. Every humanoid stack already uses one. That is common practice, not a novel idea for this project.
- I had no evidence that neuromorphic hardware helps anywhere. It does, measured, for small reactive controllers (27–53 µJ per inference). It does not for language.

### 12.9 Unverified in this section

- Figure Helix details, Genie 2/3 and Gemini Robotics On-Device come from vendor blogs read through a summarising fetch. Gemini On-Device's benchmark numbers came only from chart summaries.
- Figure 03 and Tesla Optimus current compute are undisclosed.
- Optimus power (100 W idle, 500 W walking) is from a 2022 press report.
- Tesla demo teleoperation reports are press-sourced. 1X's own page documents remote piloting and supervision.
- Agility Digit battery figures conflict across secondary sources.
- The Jetson Thor 70 W and 90 W modes are not confirmed in NVIDIA docs.
- NVIDIA's Thor tokens/s figures state no batch size or harness.
- V-JEPA 2-AC's robot table rests on 10 trials per cell with no intervals.
- SmolVLA's CPU deployment is claimed, not measured.

---

## 13. Red-team findings: prior work this survey missed

Two adversarial reviews ran after the plan was written. One hunted prior art; the other attacked logic and design. The full design critique is folded into `plan.md` v2. Below is the prior work, with the 11 key quotes I checked word for word in the downloaded papers (**[C✓]**). The survey missed the whole model-editing and memory-editing literature.

| Work | What it shows | Tag |
|---|---|---|
| Sparse memory finetuning (Lin et al., Meta, 2025), https://arxiv.org/abs/2510.15103 | Memory-layer model, updating only slots highly activated by new facts. NaturalQuestions F1 "drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop". 1.3B model; no seed spread reported; no replay baseline. | [C✓] |
| Improving SMF (2026), https://arxiv.org/abs/2604.05248 | Open pipeline to retrofit Qwen-2.5-0.5B with sparse memory, "enabling effective continual learning on consumer hardware" | [C✓] |
| SMF vs LoRA (ICML 2026 workshop), https://arxiv.org/abs/2605.03229 | SMF forgets less but learns less: +2.5 pp on MedMCQA vs +4.6 (LoRA) and +5.4 (full finetuning); seed spreads reported | [C] |
| Larimar (IBM, ICML 2024), https://arxiv.org/abs/2403.11901 | "one-shot updates of knowledge without the need for computationally expensive re-training"; 1-NN scope detector, EER 2.9% | [C✓] / [C] |
| GRACE (NeurIPS 2023), https://arxiv.org/abs/2211.11031 | Discrete key-value codebook; "Each key has a deferral radius ϵ" that gates the edit | [C✓] |
| WISE (NeurIPS 2024), https://arxiv.org/abs/2405.14768 | Routing to edit memory; says GRACE "fails to generalize" to paraphrases | [C] |
| WikiBigEdit (2025), https://arxiv.org/abs/2503.05683 | "RAG vastly outperforms specialized knowledge editing techniques"; continual LoRA beats editing at equal inference cost; WISE decays within 10K updates | [C✓] |
| LMLM (Cornell 2025), https://arxiv.org/abs/2505.15962 | 176M to 382M models with facts in an external database; "out-of-scope knowledge trigger detectable lookup failures"; below similarity 0.6 "we return unknown". Abstention is not evaluated. | [C✓] |
| SEAT (2026), https://arxiv.org/abs/2506.14387 | "standard fine-tuning often erodes aligned epistemic abstention"; sparse tuning + KL restores it | [C✓] |
| InnerExpert (2026), https://arxiv.org/abs/2608.17687 | MoE routing signals detect hallucination, answer-level AUROC up to 0.91; no single signal is uniformly strong across models | [C] |
| O'Neill (2026), https://arxiv.org/abs/2607.11020 | No tested intervention "keeps earlier facts reachable" when writing facts into weights; "the reliable channel is context" | [C✓] |
| Choi (2026), https://arxiv.org/abs/2609.03900 | A method's "5.0-point advantage over rank-8 replay becomes an 11.6-point deficit against rank-72 replay" | [C✓] |
| MemoGuard (2026), https://arxiv.org/abs/2607.15589 | On a robot: high-similarity but execution-invalid "memory traps" | [C✓] |
| Memory3, MemoryLLM, M+, Titans, KARLA, RECIPE, Memory Attention | Knowledge externalised to explicit or CPU/SSD memory; all compute on GPU. Titans Revisited: Titans "does not always outperform established baselines". | [C] |

**Effect on the survey's conclusions**
- §5 finding 1 (memory cannot replace compute) stands.
- The "open gap" in §4.5 narrows to this: no paper found evaluates a trained memory layer's lookup as an abstention signal (AUROC) against strong baselines. This rests on about 20 searches, not an exhaustive review.
- §4.2's "no memory-layer LM benchmarked on CPU-only inference" was not refuted.
- "Four independent teams" in §5 overstates independence: PKM and Memory Layers at Scale are both Meta work [R].

---

## Appendix A. Measurement scripts

Run on 2026-09-24 under load (see section 2). Python 3.12.3, numpy 2.5.1.

`membw.py`: synchronised RAM read bandwidth. Command: `python3 membw.py`.

```python
import time
import numpy as np
from multiprocessing import Barrier, Pool

N_BYTES = 1 << 30
REPEATS = 5

barrier = None


def init(b):
    global barrier
    barrier = b


def worker(_):
    a = np.ones(N_BYTES // 8, dtype=np.float64)
    a.sum()
    spans = []
    for _ in range(REPEATS):
        barrier.wait()
        t0 = time.time()
        a.sum()
        spans.append((t0, time.time()))
    return spans


if __name__ == "__main__":
    print(f"array per worker: {N_BYTES / 2**30:.1f} GiB, {REPEATS} synchronised rounds", flush=True)
    for workers in (1, 2, 4, 8):
        b = Barrier(workers)
        with Pool(workers, initializer=init, initargs=(b,)) as p:
            res = p.map(worker, range(workers))
        rounds = []
        for r in range(REPEATS):
            start = min(w[r][0] for w in res)
            end = max(w[r][1] for w in res)
            rounds.append(workers * N_BYTES / (end - start) / 1e9)
        print(
            f"workers={workers}  aggregate GB/s per round={[round(x, 1) for x in rounds]}  best={max(rounds):.1f}",
            flush=True,
        )
```

`ssdrand.py`: random 4 KiB O_DIRECT reads. The test file was made with `dd if=/dev/urandom of=ssdtest.bin bs=64M count=64 oflag=direct`, and sequential read measured with `dd if=ssdtest.bin of=/dev/null bs=64M iflag=direct`. Command: `python3 ssdrand.py`.

```python
import mmap
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

PATH = "ssdtest.bin"
BLOCK = 4096
READS_PER_THREAD = 5000

size = os.path.getsize(PATH)
n_blocks = size // BLOCK


def run(seed):
    fd = os.open(PATH, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, BLOCK)
    rng = random.Random(seed)
    lat = []
    for _ in range(READS_PER_THREAD):
        off = rng.randrange(n_blocks) * BLOCK
        t = time.perf_counter()
        os.preadv(fd, [buf], off)
        lat.append(time.perf_counter() - t)
    os.close(fd)
    return lat


print(f"file {size / 2**30:.1f} GiB, {BLOCK} B random reads, O_DIRECT, {READS_PER_THREAD} per thread", flush=True)
for threads in (1, 4, 16, 32):
    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        lats = [x for r in ex.map(run, range(threads)) for x in r]
    wall = time.perf_counter() - t
    lats.sort()
    n = len(lats)
    print(
        f"threads={threads:2d}  IOPS={n / wall:8.0f}  MB/s={n * BLOCK / wall / 1e6:6.1f}  "
        f"lat p50={lats[n // 2] * 1e6:6.0f}us  p99={lats[int(n * 0.99)] * 1e6:6.0f}us",
        flush=True,
    )
```

## Appendix B. Primary sources

**Capabilities**
- McGrew 2009, http://www.iapsych.com/articles/mcgrew2009.pdf
- Hendrycks et al. 2025, https://arxiv.org/abs/2510.18212
- Morris et al., https://arxiv.org/abs/2311.02462
- Legg & Hutter, https://arxiv.org/abs/0712.3329
- Chollet, https://arxiv.org/abs/1911.01547
- ARC-AGI-2, https://arxiv.org/abs/2505.11831
- Lake et al., https://arxiv.org/abs/1604.00289
- Botvinick et al., https://arxiv.org/abs/1711.08378
- Soar, https://arxiv.org/abs/2205.03854
- Kotseruba & Tsotsos, https://arxiv.org/abs/1610.08602
- CoALA, https://arxiv.org/abs/2309.02427
- Levy & Calvert 2021 (PNAS)
- Lennie 2003
- Azevedo 2009 (PMID 19226510)
- Pakkenberg 2003 (PMID 12543266)
- Koriat 1993 (PMID 8255951)

**Capacity**
- Allen-Zhu & Li, https://arxiv.org/abs/2404.05405
- Morris et al., https://arxiv.org/abs/2505.24832
- Kaplan, https://arxiv.org/abs/2001.08361
- Chinchilla, https://arxiv.org/abs/2203.15556
- Besiroglu, https://arxiv.org/abs/2404.10102
- Dettmers, https://arxiv.org/abs/2212.09720
- Kumar, https://arxiv.org/abs/2411.04330
- SparseGPT, https://arxiv.org/abs/2301.00774
- Wanda, https://arxiv.org/abs/2306.11695
- Densing Law, https://arxiv.org/abs/2412.04315
- Kandpal, https://arxiv.org/abs/2211.08411
- Mallen, https://arxiv.org/abs/2212.10511
- Bartol, https://elifesciences.org/articles/10778

**Memory**
- Bricken & Pehlevan, https://arxiv.org/abs/2111.05498
- Bricken 2023, https://arxiv.org/abs/2303.11934
- Krotov & Hopfield, https://arxiv.org/abs/1606.01164
- Ramsauer, https://arxiv.org/abs/2008.02217
- Kleyko, https://arxiv.org/abs/2112.15424
- PKM, https://arxiv.org/abs/1907.05242
- Memory Layers at Scale, https://arxiv.org/abs/2412.09764
- PEER, https://arxiv.org/abs/2407.04153
- UltraMem, https://arxiv.org/abs/2411.12364
- UltraMemV2, https://arxiv.org/abs/2508.18756
- Engram, https://arxiv.org/abs/2601.07372
- kNN-LM, https://arxiv.org/abs/1911.00172
- Xu 2023, https://arxiv.org/abs/2301.02828
- Wang 2023, https://arxiv.org/abs/2305.14625
- Drozdov, https://arxiv.org/abs/2210.15859
- RETRO, https://arxiv.org/abs/2112.04426
- Norlund, https://arxiv.org/abs/2302.12128
- NVIDIA RETRO, https://arxiv.org/abs/2304.06762
- Memorizing Transformers, https://arxiv.org/abs/2203.08913
- Atlas, https://arxiv.org/abs/2208.03299
- Switch, https://arxiv.org/abs/2101.03961
- LLM in a flash, https://arxiv.org/abs/2312.11514
- PowerInfer, https://arxiv.org/abs/2312.12456

**Brain-inspired**
- Cui 2016, https://arxiv.org/abs/1512.05463
- Wu & Keogh, https://arxiv.org/abs/2009.13807
- Monty, https://arxiv.org/abs/2507.04494
- SpikeGPT, https://arxiv.org/abs/2302.13939
- SpikeBERT, https://arxiv.org/abs/2308.15122
- SpikeLM, https://arxiv.org/abs/2406.03287
- SpikingBrain, https://arxiv.org/abs/2509.05276
- Loihi 2 LLM, https://arxiv.org/abs/2503.18002
- Forward-Forward, https://arxiv.org/abs/2212.13345
- Bartunov, https://arxiv.org/abs/1807.04587
- DFA, https://arxiv.org/abs/2006.12878
- PC along graphs, https://arxiv.org/abs/2006.04182
- PC critique, https://arxiv.org/abs/2304.02658
- PCX, https://arxiv.org/abs/2407.01163
- EP ImageNet, https://arxiv.org/abs/2606.03584
- van de Ven, https://arxiv.org/abs/1904.07734
- Loss of plasticity, https://arxiv.org/abs/2306.13812
- Continual pre-training, https://arxiv.org/abs/2403.08763
- DishBrain (PMC9747182)
- Balci et al. 2023 (Neuron 111:604)

**Efficient architectures**
- BitNet b1.58, https://arxiv.org/abs/2402.17764
- 2B4T, https://arxiv.org/abs/2504.12285
- bitnet.cpp, https://arxiv.org/abs/2410.16144
- Spectra, https://arxiv.org/abs/2407.12327
- Reloaded, https://arxiv.org/abs/2407.09527
- MatMul-free, https://arxiv.org/abs/2406.02528
- Mamba, https://arxiv.org/abs/2312.00752
- Mamba-2, https://arxiv.org/abs/2405.21060
- NVIDIA Mamba, https://arxiv.org/abs/2406.07887
- Jamba, https://arxiv.org/abs/2403.19887
- Repeat After Me, https://arxiv.org/abs/2402.01032
- Zoology, https://arxiv.org/abs/2312.04927
- RWKV-7, https://arxiv.org/abs/2503.14456
- xLSTM, https://arxiv.org/abs/2405.04517
- TinyStories, https://arxiv.org/abs/2305.07759
- llama.cpp discussion #4167

**Hallucination**
- Kalai et al. 2025, https://arxiv.org/abs/2509.04664
- Kalai & Vempala, https://arxiv.org/abs/2311.14648
- Gekhman, https://arxiv.org/abs/2405.05904
- Huang survey, https://arxiv.org/abs/2311.05232
- Kadavath, https://arxiv.org/abs/2207.05221
- Semantic entropy, https://www.nature.com/articles/s41586-024-07421-0
- SelfCheckGPT, https://arxiv.org/abs/2303.08896
- Azaria & Mitchell, https://arxiv.org/abs/2304.13734
- Levinstein & Herrmann, https://arxiv.org/abs/2307.00175
- R-Tuning, https://arxiv.org/abs/2311.09677
- Abstention survey, https://arxiv.org/abs/2407.18418
- SimpleQA, https://arxiv.org/abs/2411.04368
- Legal RAG, https://arxiv.org/abs/2405.20362
- Lost in the Middle, https://arxiv.org/abs/2307.03172
- RGB, https://arxiv.org/abs/2309.01431
- Sufficient Context, https://arxiv.org/abs/2411.06037
- Self-RAG, https://arxiv.org/abs/2310.11511
- Adaptive retrieval, https://arxiv.org/abs/2501.12835
- GrailQAbility, https://arxiv.org/abs/2212.10189
- Long-tail kNN-LM, https://arxiv.org/abs/2503.22426
- Magnitude Mirage, https://arxiv.org/abs/2609.15578

**Robot brain (section 12)**
- NVIDIA Jetson Thor, https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
- NVIDIA Jetson Orin, https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- 1X NEO, https://1x.tech/neo
- Figure 03 battery, https://www.figure.ai/news/f-03-battery-development
- Figure Helix, https://www.figure.ai/news/helix
- Figure Helix 02, https://www.figure.ai/news/helix-02
- Unitree G1, https://www.unitree.com/g1
- Deniz et al. 2026 (G1 power), https://arxiv.org/abs/2606.15915
- Liu, Shi & Shin (robot power breakdown), https://arxiv.org/abs/2511.20467
- Levy & Calvert 2021, PNAS 118(18) e2008173118
- Kurtzer 2015 (PMC4310276)
- Mini-Cheetah control, https://arxiv.org/abs/1909.06586
- π real-time chunking, https://arxiv.org/abs/2506.07339
- RT-2, https://arxiv.org/abs/2307.15818
- OpenVLA, https://arxiv.org/abs/2406.09246
- π0, https://arxiv.org/abs/2410.24164
- π0.5, https://arxiv.org/abs/2504.16054
- π*0.6 / RECAP, https://arxiv.org/abs/2511.14759
- Gemini Robotics, https://arxiv.org/abs/2503.20020
- Gemini Robotics 1.5, https://arxiv.org/abs/2510.03342
- GR00T N1, https://arxiv.org/abs/2503.14734
- SmolVLA, https://arxiv.org/abs/2506.01844
- TinyVLA, https://arxiv.org/abs/2409.12514
- Jetson-PI, https://arxiv.org/abs/2607.12659
- LIBERO-Plus, https://arxiv.org/abs/2510.13626
- LIBERO-PRO, https://arxiv.org/abs/2510.03827
- I-JEPA, https://arxiv.org/abs/2301.08243
- V-JEPA, https://arxiv.org/abs/2404.08471
- V-JEPA 2, https://arxiv.org/abs/2506.09985
- DUET-DINO, https://arxiv.org/abs/2609.10506
- DINO-WM, https://arxiv.org/abs/2411.04983
- DreamerV3, https://arxiv.org/abs/2301.04104
- Dreamer 4, https://arxiv.org/abs/2509.24527
- WEAVER, https://arxiv.org/abs/2606.13672
- LIBERO, https://arxiv.org/abs/2306.03310
- VLA forgetting, https://arxiv.org/abs/2603.03818
- Real-world continual VLA, https://arxiv.org/abs/2605.26820
- Simple Recipe Works, https://arxiv.org/abs/2603.11653
- KnowNo, https://arxiv.org/abs/2307.01928
- Introspective Planning, https://arxiv.org/abs/2402.06529
- SAFE, https://arxiv.org/abs/2506.09937
- Neuromorphic drone, https://arxiv.org/abs/2303.08778
- Neuromorphic force control, https://arxiv.org/abs/2403.08928
