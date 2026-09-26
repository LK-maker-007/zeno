# Paths to Ava: who is building a general mind outside the Transformer, and what the evidence rules out

Date: 2026-09-26. Companion to `foundational-survey.md`, `agi-asi-survey.md` and `plan.md`. It covers what those left out:

- programs building general intelligence outside the Transformer, including the ones that failed;
- novel-problem reasoning with small systems (ARC);
- learning from a raw stream of experience;
- the arithmetic behind the goal in `plan.md` §1.1: a brain that runs on this laptop.

Tags:
- **[V]** measured on this laptop, or read directly from code, a checkpoint, data or a leaderboard, for this document.
- **[C]** a primary source says it; the link is given. Every [C] number in sections 3 to 6 was read in its source twice: once in the census and again in a separate verification pass. The section 9 checks were read once.
- **[S]** press, a vendor page or a search snippet only.
- **[H]** hypothesis, or my arithmetic with the inputs shown.
- **[R]** from memory, unchecked.

Laptop timings are provisional. Only the section 6.1 benchmark ran on an idle machine; the others ran while other jobs used the CPU.

---

## 1. The decision

**Question.** Does any known architecture beat `plan.md` v3 as the road to the §1.1 goal? That goal is a mind of our own that runs on a laptop CPU, learns on the job from what it sees and hears, answers only what it knows, and fits in bounded memory.

**Threshold, set before the census.** Switch only to a direction that has all three:
1. an independently verified result on a general-capability test, against a baseline;
2. measured CPU inference;
3. demonstrated learning on the job without forgetting.

**Current setting.** `plan.md` v3: the C1 child-mind, Design D memory, the A1 fast-weight reader, the scoreboard and the ARC-AGI-3 arena.

**Answer.** No direction has all three. None has (1) and (3) together. The plan stands; section 8 lists what the census adds to it.

## 2. The goal's requirements against the numbers

| Requirement (`plan.md` §1.1) | Verdict | Evidence |
|---|---|---|
| Brain-like power, about 20 W | Supported, as power | i5-11300H: "Configurable TDP-down 28 W", "Configurable TDP-up 35 W" [C, [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/196656/intel-core-i511300h-processor-8m-cache-up-to-4-40-ghz-with-ipu/specifications.html)] |
| Emulate the brain's neurons on this laptop | Ruled out | 26x to 259,000x short on compute; at least 3,600x short on memory (§6) [H] |
| A different algorithm reaching human function inside this laptop | Undecided | Carlsmith: "None of these methods are direct guides to the minimum possible FLOP/s budget" [C] |
| No memory constraint | Impossible | Optimal continual learning algorithms "generally solve an NP-hard problem and will require perfect memory to do so" [C, [Knoblauch et al. 2020](https://proceedings.mlr.press/v119/knoblauch20a.html)] |
| Never hallucinate | Possible only by answering less | A question-answer database that otherwise says "I don't know" does not hallucinate, and gives up coverage [C, `foundational-survey.md` §4.5] |

---

## 3. Programs building general intelligence outside the Transformer

"Demonstrated" means a number on a benchmark against a baseline. A claim with no such number is listed as a claim.

| Program | What it has shown | Compute | Status 2026 |
|---|---|---|---|
| **Thousand Brains Project, Monty** | 77 YCB objects × 14 rotations: "Monty requires approximately 528,000,000× fewer FLOPs for training" than the pretrained ViT that matches its accuracy; "approximately 4 million parameters" [C, [arXiv 2507.04494](https://arxiv.org/abs/2507.04494) §4.3]. "Inference FLOPs scales linearly with the number of known models" [C, same]. | "All experiments were run on 16 CPUs with parallelization"; 77-object runs 12 to 37 min [C, [docs](https://docs.thousandbrains.org/docs/benchmark-experiments), v0.51.0] | Active. MIT licence, repo created 2024-11-13, v0.51.0 on 2026-09-24 [V, GitHub API] |
| **Keen Technologies** | Physical Atari robot: response time "roughly 165 ms"; six games, 600,000 steps at 30 fps (5.5 h); replay with batch 16. Policies "consistently performed worse when tested on the Robotroller that was not used for learning" [C, [arXiv 2606.19357](https://arxiv.org/abs/2606.19357)] | "We have been happy with Lambda Labs" [C, [Carmack talk notes](https://docs.google.com/document/d/1-Fqc6R6FdngRlxe9gi49PRvU97R83O7ZTN_KFXo_jf0)] | Active |
| **Oak Lab** (Sutton, Javed) | Claim: "A trillion-parameter agent that learns and plans in real-time with 20 watts of power" [C, [oaklab.ai](https://oaklab.ai/mission.html)]. Shown: one NoisyMNIST figure [C] | None reported | First post 13 Jul 2026 [C, feed]. The founders leaving Keen is press only [S] |
| **AMI Labs** (LeCun) | Nothing published: no paper, model or code on its site, Hugging Face or GitHub [V] | "We've raised a $1.03B USD (~€890M) round" [C, [amilabs.xyz](https://amilabs.xyz/updates)] | Launched 10 Mar 2026 [C] |
| **VERSES AXIOM / Genius** (active inference) | Its text claims AXIOM beats BBF and DreamerV3 "in every Gameworld environment". Its own Table 1 puts AXIOM last on Cross (−68 vs −48 and −27) and Drive (−49 vs −37 and −45) [C, [arXiv 2505.24784](https://arxiv.org/abs/2505.24784)]. Removing the information-gain term gives a higher mean reward on 8 of 10 games [V, Table 1] | "About 30min" per game on one A100. Table 2's per-step times give 45 to 92 min [C/H] | "The Company intends to discontinue its artificial intelligence ("AI") research and development activities"; Friston resigned. 18 Jun 2026 [C, [SEC exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1879001/000149315226029568/ex99-1.htm)] |
| **HRM** (Sapient) | Claimed 40.3% on ARC-AGI-1 with 27M parameters [C, [arXiv 2506.21734](https://arxiv.org/abs/2506.21734)]. ARC Prize re-ran it: 32% ARC-AGI-1, 2% ARC-AGI-2. "A regular transformer comes within ~5pp"; the "outer loop" refinement "drove substantial performance" [C, [ARC Prize](https://arcprize.org/blog/hrm-analysis)] | $148.50 for the ARC-AGI-1 run; $201 for ARC-AGI-2 [C] | Superseded by TRM (§4) |
| **Sakana CTM** | ImageNet "72.47% top-1" with a modified ResNet-152 backbone [C, [arXiv 2505.05522](https://arxiv.org/abs/2505.05522)] | "8 H100" [C] | No follow-up found |
| **Liquid LFM2** | A hybrid, not attention-free: 6 of 16 blocks are attention in the 350M, 700M and 1.2B models. The 1.2B decodes 99.7 tok/s on CPU (llama.cpp, Q4_0, batch 1, 1K prefix) [C, [arXiv 2511.23404](https://arxiv.org/abs/2511.23404) Tables 1, 3] | "Pre-trained on 10–12T tokens" [C] | Shipping |
| **Pathway BDH** | "BDH rivals GPT2-architecture Transformer performance" from 10M to 1B parameters [C, [arXiv 2509.26507](https://arxiv.org/abs/2509.26507)]. Its README: the code "does not reproduce the 97.4% benchmark result out of the box" [C, [repo](https://github.com/pathwaycom/bdh)] | GPU | Open code |
| **ONA** (NARS) | Q-learning "performed better on CliffWalking-v0, Taxi-v3, and FlappyBird-v0"; ONA "more promising" on FrozenLake [C, [arXiv 2304.03291](https://arxiv.org/abs/2304.03291)] | Tiny | Research |
| **AERA** | A 100-word interview task, "after an observation period of approximately 20 hours, has been correctly learned" [C, [arXiv 1312.6764](https://arxiv.org/abs/1312.6764)] | Not stated | 2013 demonstration |
| **Spaun** | "2.5-million-neuron model of the brain", eight tasks; "the model is unable to learn completely new tasks" [C, [Eliasmith 2012](https://compneuro.uwaterloo.ca/files/publications/eliasmith.2012.pdf)] | About 2.5 h of computer time per simulated second [S, press] | Dormant [H] |
| **OpenCog Hyperon** | No benchmark result found | n/a | Research |

### The failures

| Program | What happened |
|---|---|
| **Cyc** | "Four decades, 2000 person-years"; "larger-sized teams generally showed a net decrease in total productivity". The general theorem prover was turned off after "over a million queries in a row that called on it, as a last resort, just timed out" [C, [arXiv 2308.04445](https://arxiv.org/abs/2308.04445)] |
| **Vicarious RCN** | On RCN's own CAPTCHA data, a gated fully-convolutional network with augmentation reached sequence error 12.30% against RCN's 33.4% (21.70% without augmentation; its scoring is case-sensitive, RCN's is not) [C, [arXiv 1812.11894](https://arxiv.org/abs/1812.11894) Table 1] |
| **Human Brain Project** | Fell short "not only because of technological limitations, but also because of the lack of new concepts" [C, [Frégnac 2023](https://www.eneuro.org/content/10/11/ENEURO.0428-23.2023)] |
| **OpenWorm** | "The current implementation can demonstrate basic locomotion" [C, secondary: [State of Brain Emulation 2025](https://arxiv.org/abs/2510.15745)] |

---

## 4. Novel-problem reasoning with small systems (ARC)

Laptop timings come from the scripts in Appendix A. The session they ran in measured fp32 matmul at 133 GFLOPS [V]; the idle benchmark in §6.1 reached 386 [V]. So these CPU times may be up to about 3x pessimistic [H].

| System | Size | Adapts to each task at test time? | Score | Training compute | This laptop |
|---|---|---|---|---|---|
| icecuber (2020 DSL search) | C++ search | No | ARC-1 17%, ARC-2 1.6%, semi-private [V, [evaluations.json](https://arcprize.org/media/data/evaluations.json)] | None | Depth 2: 2 min 42 s for 419 test inputs; 120/419 top-2 (28.6%), 129/419 top-3, matching the README's 129 [V] |
| CompressARC | "76K parameters" [C], plus 69,424 to 2,190,840 per-puzzle latent values [V] | Yes, from scratch, no pretraining | ARC-1 eval 20.00% at 2000 steps, 6.00% at 200 [C, [arXiv 2512.06104](https://arxiv.org/abs/2512.06104) Table 5]. ARC-2 4% [C, secondary: [arXiv 2601.10904](https://arxiv.org/abs/2601.10904)] | None | 1.253 s/step median over 16 tasks, so 41.8 min per puzzle [V] |
| HRM | 27M | Yes: its training set includes the evaluation tasks' demonstrations [C] | 32% / 2% verified [C] | $148.50 for the ARC-1 run [C] | Not measured |
| TRM | 6,830,082 trunk plus a 448,719,872-parameter puzzle-embedding table [V, [checkpoint](https://huggingface.co/arcprize/trm_arc_prize_verification)] | Yes, as HRM | 40% / 6.25%, semi-private [V, evaluations.json] | "Around 3 days with 4 H100" [C, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871)] | 38.88 s per augmented pass; 1000 passes take 10.8 h per test input [V] |
| URM | 4 layers, width 512 | Yes | 53.8% pass@1 ARC-1, 16.0% ARC-2, self-reported [C, [arXiv 2512.14693](https://arxiv.org/abs/2512.14693)] | Not stated | Not measured |
| VARC | 18M ViT | Yes, fine-tuned per task | 54.5±0.7 ARC-1, 8.3±0.4 ARC-2 over four runs; ensemble 60.4 / 11.1 [C, [arXiv 2511.14761](https://arxiv.org/abs/2511.14761)] | 8×H100 for 4.8 h; "70 seconds per task on a single GPU" [C] | 34.22 s per batch-8 step; one 2000-step fine-tune is 19.0 h [V/H] |
| LPN | 178M | Latent search only, no weight updates | 15.25% ARC-1 eval at 2e13 FLOPs per task [C, [arXiv 2411.08706](https://arxiv.org/abs/2411.08706)] | "2 days on a TPU v4-32" [C] | About 2.6 min per task at that session's 130 GFLOP/s; about 52 s at the idle 386 [H, 2e13 ÷ FLOP/s] |
| ARC-NCA | Tiny cellular automata | Yes, from scratch | 6.5 to 12.9% per model, 17.6% for the union, on only the 262 same-size tasks [C, [arXiv 2505.08778](https://arxiv.org/abs/2505.08778)] | None | Not measured |
| VSA solver | Symbolic | Search | 3.0% ARC-1, 0.0% ARC-2 [C, [arXiv 2511.08747](https://arxiv.org/abs/2511.08747)] | None | Not measured |
| StochasticGoose (ARC-AGI-3 preview winner) | 34,320,614-parameter CNN [V] | Yes, re-created on each new level [C, code] | 12.58%, 255,964 actions [C, [ARC Prize](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)] | None | 1.79 s per action, so about 16,000 actions in 8 h [V/H] |

For scale:
- The 2025 Kaggle ARC-AGI-2 winners were language models: NVARC 24.03% with Qwen3-4B and 103k synthetic puzzles; the ARChitects 16.53% with LLaDA-8B [C, [arXiv 2601.10904](https://arxiv.org/abs/2601.10904)].
- "ARC-AGI-2 accuracies below 5% are generally not treated as meaningful" [C, [arXiv 2505.11831](https://arxiv.org/abs/2505.11831)].

**ARC-AGI-3 standings** [V, Kaggle leaderboard CSV of 2026-09-25]:
- 3,327 teams; median score 0.31.
- Top three: Lord Han Solo 19.45, Tufa Labs 18.81, Daniel Franzen 16.68.
- The best entry I could confirm is LLM-free is a BFS solver at 0.30. ARC Prize wrote that the preview winners explored "in the hope of encountering a winning combination by chance" [C, [arXiv 2603.24621](https://arxiv.org/abs/2603.24621)].
- Every public game is "reachable through non-intelligent strategies" [C, [arXiv 2605.25931](https://arxiv.org/abs/2605.25931)].

---

## 5. Learning from a raw stream of experience

| Work | Data | One pass? | Result against baseline | Compute |
|---|---|---|---|---|
| Stream-Q / Stream-AC ([arXiv 2410.14606 v3](https://arxiv.org/abs/2410.14606)) | 200M frames per Atari game | Yes, no buffer | Median 0.99 vs 0.67 for DQN with a 10^6-transition buffer (Atari). Stream-AC 0.73 vs 1.00 for the per-task best of PPO and SAC (50 control tasks). With a one-transition buffer: DQN 0.00, SAC 0.04, PPO 0.07 [C] | Not extracted |
| Stream Q(λ) as a baseline ([arXiv 2602.09396](https://arxiv.org/abs/2602.09396)) | 26 Atari games | Yes | Human-normalised IQM 0.23 at 40M frames, 0.44 at 100M; 5 seeds [C] | 19.52 h for 40M frames on 4 CPU cores; 1.68M parameters [C] |
| AVG ([arXiv 2411.15370](https://arxiv.org/abs/2411.15370)) | Real robots | Yes | "SAC-100 and SAC-1 struggle significantly, failing to learn under the imposed memory limitations" [C] | Jetson Nano 4GB, "about 37ms per update" (mobile-robot task) [C] |
| Keen physical Atari (§3) | 5.5 h per game | **No: replay, batch 16** | Worse on a second, identical body [C] | GPU |
| Loss of plasticity ([Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)) | 2,000 tasks | No | Up to 88% on early tasks, then "by the 2,000th task, they had lost substantial plasticity"; the preprint gives 89% to 77% [C]. Continual backprop maintains plasticity "apparently indefinitely" [C] | Not stated |
| CVCL ([Science 2024](https://doi.org/10.1126/science.adi1374)) | 61 h of one child's headcam | No, "up to 400 epochs" | 61.6% vs CLIP 66.7% on 22 concepts, four-way choice, chance 25%. On 64 object categories 34.7% vs CLIP 99.4% [C] | Not stated |
| Orhan &amp; Lake ([arXiv 2305.15372](https://arxiv.org/abs/2305.15372)) | 472 h of headcam | No | "65-70% of a model trained on the full ImageNet training set", comparable to training on 10% of ImageNet [C] | "Four days on four A100 GPUs" per model [C] |
| BabyView ([arXiv 2406.10447](https://arxiv.org/abs/2406.10447)) | 868 h of headcam | No | ImageNet linear probe 53.28 vs 77.64 for ImageNet training; 430 h and 868 h both give 53.28. Matching DINOv2 "would require upwards of 10^7 hours" [C] | Not extracted |
| BabyLM 2023, 2024 ([2023](https://aclanthology.org/2023.conll-babylm.1.pdf), [2024](https://arxiv.org/abs/2412.05149)) | 10M / 100M words | No: the 2023 winner ran "over 450 epochs" (Strict) and "over 2000 epochs" (Strict-Small) | 2024: GPT-BERT BLiMP 86.1; EWoK "maximum score was 58.4%" (chance 50%); "No submissions outperformed the baselines in the multimodal track"; "a strong relationship between training FLOPs and average performance" [C] | Small |
| STELA raw speech ([arXiv 2306.01506](https://arxiv.org/abs/2306.01506)) | 1024 h of child-centred audio | No | Lexical accuracy 49.5 vs 49.2 for random [C] | Not found |
| Titans Revisited ([arXiv 2510.09551](https://arxiv.org/abs/2510.09551)) | | Within one sequence | "Memory updates alone are insufficient for meaningful test-time learning", with a frozen backbone [C] | GPU |
| TTT-E2E ([arXiv 2512.23675](https://arxiv.org/abs/2512.23675)) | | Within one sequence | "Transformer with full attention dramatically outperforms the other methods, including ours, especially in long context" [C] | GPU |
| Nested Learning / HOPE ([arXiv 2512.24695](https://arxiv.org/abs/2512.24695)) | Llama3-8B plus 15B tokens of continual pre-training | Within one sequence | Continual-learning results only in figures. In 22 of 90 citing papers checked, every one reporting HOPE numbers has a HOPE author [C/V] | GPU |
| WSCL ([arXiv 2401.08623](https://arxiv.org/abs/2401.08623)) | CIFAR-10 | No | ER-ACE 59.98 to 71.15, both with a 200-sample buffer; WSCL adds a 5,000-sample short-term buffer [C] | Small |
| SRC ([Nat. Commun. 2022](https://www.nature.com/articles/s41467-022-34938-7)) | Incremental MNIST | No | Sequential 19.49, SRC 48.47 ± 5.03 [C] | Small |
| SIESTA ([arXiv 2303.10725](https://arxiv.org/abs/2303.10725)) | ImageNet | No | Only the first 8 layers (2.19% of parameters) are frozen; "the remaining 11 layers ... (97.81%) are trained during sleep" on stored latents [C] | Small |

Two further results on plasticity:
- Plasticity loss also appears in GPT-style models "ranging from 5M to 314M non-embedding parameters" [C, [arXiv 2606.24752](https://arxiv.org/abs/2606.24752)].
- In a comparison of fixes, "of these methods, ReDo performs the worst" [C, [arXiv 2405.19153](https://arxiv.org/abs/2405.19153)].

**The human-scale gap.** Children hear "less than 100 million word tokens by age 13" [C, BabyLM 2023, citing Gilkerson et al. 2017].
- At that budget, grammar is learned: 86.1 on BLiMP.
- World knowledge is not: at most 58.4% on EWoK, against 50% chance.
- One child's 61 h of video gives 61.6% against CLIP's 66.7%, and only on concepts from that child's own world.

---

## 6. The brain against this laptop

### 6.1 This laptop, measured idle

Load average 0.35 before the run. Script in Appendix A.1. numpy 2.5.1 with OpenBLAS 0.3.33 (the build reports a Haswell kernel, so it may not use AVX-512). One run of 5 repeats per setting [V].

| Threads | fp32 matmul 4096², best of 5 | Batch-1 layer d=1024 | Batch-1 layer d=2048 |
|---|---|---|---|
| 1 | 123.8 GFLOP/s | 24.4 GFLOP/s | 13.9 GFLOP/s |
| 4 | 386.1 GFLOP/s (runs 264.5 to 386.1) | 72.1 GFLOP/s | 25.6 GFLOP/s |
| 8 | 359.6 GFLOP/s | 73.2 GFLOP/s | 18.3 GFLOP/s |

A d=1024 layer (4 MB) fits in the "8M Cache" [C, [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/196656/intel-core-i511300h-processor-8m-cache-up-to-4-40-ghz-with-ipu/specifications.html)]; a d=2048 layer (16.8 MB) does not.

### 6.2 Brain numbers

| Quantity | Value | Source |
|---|---|---|
| Brain FLOP/s, mechanistic method | "10^13-10^17 FLOP/s is enough ... seem plausible to me"; 10^15 "more likely than not" enough; "unlikely (<10%) that more than 10^21 FLOP/s is required" | [C, [Carlsmith 2020](https://web.archive.org/web/20241230013507/https://www.openphilanthropy.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/) §1.1] |
| Brain FLOP/s, all methods | Functional method from the retina: 1e12 to 1e15. Limit method: 7e21 | [C, Carlsmith §6] |
| Whole Brain Emulation roadmap | Level 3 (population model) 10^15 FLOPS; level 4 (spiking network) 10^18 FLOPS and 8 bytes per synapse; level 10 10^43 | [C, [Sandberg &amp; Bostrom 2008](https://web.archive.org/web/20201116130201/http://www.fhi.ox.ac.uk/brain-emulation-roadmap-report.pdf) Tables 8, 9] |
| Human brain, simple spiking neurons (LIF), 10 Hz | Event-driven 3.4e16 FLOP/s; time-stepped 8.5e18; state 1.36 PB. The report's upper bounds assume multi-compartment Hodgkin-Huxley neurons | [C, [State of Brain Emulation 2025](https://arxiv.org/abs/2510.15745), [data](https://github.com/MxSchons-GmbH/sobe-2025-data-repository/blob/6e8fd472308cd958cbbf50ca4339b7b3da6f6815/data/compute/computational-demands-organisms.tsv)] |
| Largest human-scale runs | 86e9 neurons on 14,012 GPUs, 65 to 118.8 times slower than real time [C, [arXiv 2308.01241](https://arxiv.org/abs/2308.01241)]. A human-scale cerebellum on all of the K computer, "578 times slower than the wall clock time" [C, [Frontiers 2020](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2020.00016/full)] | |
| Synapses | Neocortex: "164 x 10(12) (CV = 0.17)", five brains [C, [Tang et al. 2001](https://pubmed.ncbi.nlm.nih.gov/11418939/)]. No direct whole-brain count found; estimates run "between 100-1000 trillion" [C, SoBE 2025] | |
| Bits per synapse | "26 distinguishable synaptic strengths, corresponding to storing 4.7 bits", rat hippocampus, 287 synapses [C, [Bartol et al. 2015](https://elifesciences.org/articles/10778)]. Later: 4.1 to 4.59 bits [C, [Samavat et al. 2024](https://doi.org/10.1162/neco_a_01659)] | |
| Functional memory | "A functional learned memory content of around a billion bits for a mature person". The storage hardware may hold "a thousand to a million times the capacity manifest in learned behavior" | [C, [Landauer 1986](https://doi.org/10.1207/s15516709cog1004_4) pp. 491, 492] |
| Fly brain model | "All 127,400 proofread neurons", over 50 million connections; "91% were consistent" across 164 tested predictions; "approximately 5 min per 1,000 ms trial per central processing unit thread", no CPU model named | [C, [Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9)] |
| Same model on Loihi 2 | 12.40 ms per simulated second on 12 chips; its Brian 2 baseline took about 4.4 s | [C, [arXiv 2508.16792](https://arxiv.org/abs/2508.16792) Table 1] |
| Energy per synaptic event | Cortex: 19 to 760 fJ. CPU cluster simulator: 4.4 µJ (5.8 µJ with the network switch) [C, [van Albada et al. 2018](https://doi.org/10.3389/fnins.2018.00291)]. SpiNNaker 2 prototype: 0.20 to 0.26 nJ [C, [Höppner et al. 2021](https://arxiv.org/abs/2103.08392) Table I] | |
| Hala Point | "Up to 1.15 billion neurons and 128 billion synapses", "a maximum of 2,600 watts", "over 380 trillion 8-bit synapses ... per second"; "not intended for neuroscience modeling" | [C, [Intel, 17 Apr 2024](https://download.intel.com/newsroom/archive/2025/en-us-2024-04-17-intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai.pdf)] |
| One neuron as a network | "A temporally convolutional DNN with five to eight layers was required" [C, [Beniaguev et al. 2021](https://doi.org/10.1016/j.neuron.2021.07.002)]; later matched "with under ten thousand trainable parameters" [C, [arXiv 2306.16922](https://arxiv.org/abs/2306.16922)] | |

### 6.3 Arithmetic [H]

Every line is arithmetic on the numbers above. The laptop figure is 386 GFLOP/s, the best idle matmul.

**Compute.**
- Carlsmith's functional low (1e12) is 2.6x the laptop.
- His mechanistic range (1e13 to 1e17) is 26x to 259,000x; his central 1e15 is 2,600x.
- Simple spiking neurons, event-driven (3.4e16), are 88,000x.
- Roadmap level 4 (1e18) is 2.6 million times.

**Memory.**
- Synaptic weights: 1e14 to 1e15 synapses × 4.7 bits = 59 TB to 0.59 PB. That is 3,600x to 36,000x the 16.5 GB of RAM, and 300x to 3,000x the 199 GB of free SSD.
- It never reaches the Salk press release's "at least a petabyte" [S]; that needs 1.7e15 synapses.
- Landauer's 1e9 bits is 125 MB, under 1% of RAM. His factor of a thousand to a million gives 125 GB (fits the SSD) to 125 TB (does not fit).

**Energy.**
- A CPU simulator spends 4.4 to 5.8 µJ per synaptic event against the cortex's 19 to 760 fJ: 5.8 million to 300 million times more.
- Hala Point at peak works out to 2,600 W ÷ 3.8e14 = 6.8 pJ per synaptic operation. That is from Intel's peak figures, not a measurement.

**Insect scale fits.**
- 50M fly synapses × 8 bytes (roadmap level 4) is 0.4 GB.
- The published model runs about 300x slower than real time per CPU thread.

**Online learning on this laptop.**
- Assume 24 bytes of memory traffic per parameter per update and 30 updates per second.
- Bandwidth then caps the model at 27M parameters at the measured 19.8 GB/s (`foundational-survey.md` §2), or 71M at the theoretical 51.2 GB/s.
- Compute is not the limit: at 6 FLOP per parameter per sample and the measured 14 to 26 GFLOP/s for a batch-1 layer that does not fit in cache, the cap is 78M to 142M.
- So a learner that updates 30 times a second is limited to roughly 25 to 70M parameters.

**Streaming RL at published scale.**
- Stream Q(λ) took 19.52 h for 40M frames on 4 CPU cores, so the 200M frames per game used in the Stream-X paper would take about 98 h per game.
- 200M frames is 926 h of play at 60 fps.

**Storing what Ava sees.**
- Raw 84×84 grey frames at 30 fps for 16 waking hours: 12.2 GB a day.
- SIESTA-size latents (1,568 bytes per image) at 1 frame per second: 90 MB a day, 33 GB a year.
- That fits the SSD, not RAM, so a rule for what to drop is required.

**Kaggle.**
- `foundational-survey.md` §7.4 assumes 65 TFLOPS [R] at 30% on 2 GPUs for 30 h, which gives 4.2e18 FLOP a week.
- Training TRM from scratch is about 1.6e20 FLOP (518,071 steps × 768 × 3.98e11), about 38 weeks of that budget.
- I-JEPA's "16 A100 GPUs in under 72 hours" [C, [arXiv 2301.08243](https://arxiv.org/abs/2301.08243)] is 1,152 A100-hours for a vision encoder alone.

**What the numbers support.**
- Power: the brain and the laptop are the same order.
- Insect-scale emulation runs on a CPU.
- Carlsmith: training "is much more resource-intensive than using it to do X once trained" [C], consistent with "GPUs only for training".

**What they contradict.**
- Emulating a human brain neuron by neuron on this laptop: compute, memory and bandwidth all fall short by orders of magnitude.
- "No TBs of memory" for any design that stores per-synapse state.

**What they cannot decide.**
- Whether a different algorithm reaches human-level function in 386 GFLOP/s and 16.5 GB.
- Carlsmith's lowest estimate is 2.6x the laptop. No method gives a floor.

---

## 7. Matrix

### What every system that worked did
1. **Adapted to the task at test time.** This holds for every neural ARC system that scored. The exception is icecuber, which reaches 17% on ARC-1 by search alone.
   - ARC Prize 2024: "there does not exist any static inference-style transduction solution that scores above 11%" [C, [arXiv 2412.04604](https://arxiv.org/abs/2412.04604)].
   - TRM with a blank or random puzzle ID scores 0.00% [C, [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)].
   - HRM trained on the evaluation tasks alone still reached 31% [C, ARC Prize].
2. **Refined iteratively.**
   - CompressARC: 6.00% at 200 steps, 20.00% at 2000.
   - LPN: 9.10% at 10 search steps, 15.50% at 400.
   - HRM: one refinement loop added 13 points [C].
3. **Reused its data.**
   - The BabyLM 2023 winner ran over 450 and over 2000 epochs; CVCL up to 400 [C].
   - The single-pass Atari successes consumed 40M to 200M frames per game [C].
4. **Kept past data.**
   - Keen's working robot learner uses replay [C].
   - Every sleep method above keeps a buffer or stored latents [C].
   - Knoblauch et al. derive why: their result "explain[s] the excellent performance of CL algorithms using experience replay, episodic memory and core sets relative to regularization-based approaches" [C].
5. **Kept attention or associative recall.**
   - TRM's checkpoint has self-attention in both layers [V]; LFM2 keeps 6 of 16 blocks [C].
   - The Zeno survey's recall finding stands: pure recurrent models fail associative recall (`foundational-survey.md` §5).
6. **Was scored by an outside benchmark, with public code.** That is how HRM's 40.3% became 32%.

### What went with failure
- **Biological detail with no behavioural target:** HBP, OpenWorm.
- **Knowledge that grew only with human labour:** Cyc.
- **Wins against weak or self-chosen baselines, later reversed:** Vicarious RCN, AXIOM's own table, HRM's 40.3%.
- **More capacity on scarce data:**
  - VARC 66M scores 53.0 against the 18M's 54.5.
  - A 16-layer, width-1024 plain transformer scores 0.00 pass@1 [C].
  - TRM: "adding layers decreased generalization" [C].
- **Streaming without stabilisers:** DQN 0.00, SAC 0.04, PPO 0.07 with a one-transition buffer [C].
- **Compute-capped reproductions:** TRM inside Kaggle limits scored 2.08% (NVARC) and 3.33 to 6.67% (McGovern) [C].
- **Money ran out:** VERSES; HBP.

### Tested and found not load-bearing
- **HRM's hierarchy:** a plain transformer comes "within ~5pp" [C].
- **HRM's adaptive halting:** "within a few percentage points" of a fixed 16 loops [C].
- **TRM's recursion at inference:** step 1 reaches 38.25% of the final 40.50% [C].
- **TRM added to an LLM ensemble:** 27.22 with and without it [C].
- **AXIOM's information-gain term:** removing it raises mean reward on 8 of 10 games [V].

### Not shown to help
- **Monty's hierarchy:** below its flat control on the clean compositional benchmark (30.83 vs 34.21), above it under noise and random rotation (23.68 vs 18.95). The docs say these benchmarks "are not currently expected to have good performance" [C, v0.51.0].

### Beliefs the census contradicted
- **Titans and HOPE are routes to lifelong learning.** They adapt within one sequence on backbones trained offline.
- **Active inference is what makes AXIOM work.** Its own ablation says otherwise.
- **More of one child's experience helps.** BabyView gives 53.28 at 430 h and at 868 h.
- **This laptop does about 130 GFLOPS.** That figure was measured under load; idle, it does 386.

---

## 8. What this changes in the plan [H]

The census does not move the plan off its road. The bet in `plan.md` §4 fits the evidence:
- explicit records with sources are the perfect memory of the past that Knoblauch et al. say optimal continual learning needs;
- a measured "I don't know" is something no program in section 3 reports;
- an own mind raced against a fair baseline is the step that caught HRM.

Additions supported by the matrix:
1. **Test-time adaptation and iterative refinement for A1's successors.** A1 reads memory twice but does not adapt to the task. Findings 1 and 2 are the two ingredients every neural ARC system that scored shares.
2. **Design D's records as the replay source for learning on the job.** Finding 4. M2 already lists replay as an opponent; it should also be a candidate mechanism.
3. **Replace "no memory constraint" with a measured budget:**
   - bytes stored per day;
   - a rule for what is dropped;
   - T2 forgetting measured under that rule.
4. **Ruled out as the main path, each for a named reason:**
   - neuron-level emulation (§6.3);
   - regularisation-only protection against forgetting (EWC: `foundational-survey.md` §4.3; Knoblauch et al.);
   - brain-inspired structure adopted without an ablation (HRM, Monty, AXIOM);
   - active inference as the core (AXIOM's ablation; VERSES shut its research);
   - Monty as a base (no language result, `foundational-survey.md` §4.3; it would also break Rule 2).

**Next measurements, cheapest first.**
1. **The A1 vs C1 race** pre-registered in `a1-designs.md`. Nothing here changes its pass or kill criteria.
2. **New, V1: one-pass vision on this CPU.**
   - Train a small self-supervised encoder on a few hours of public egocentric video, once in a single pass and once for several epochs on the same frames, and compare their linear probes. Five seeds.
   - **Kill, proposed before running:** if the upper bound of the 95% CI, over 5 seeds, of the ratio (one-pass probe / multi-epoch probe) is below 0.8, one-pass perception learning is dropped. Ava then learns perception by replaying stored records.
   - This tests the "learns everything it sees" requirement directly. The census found no one-pass result at human scale.

---

## 9. A secondary summary, checked claim by claim

A summary of this field was circulating. Most of its facts hold; three would mislead a build decision.

| Claim | Verdict |
|---|---|
| bitnet.cpp runs a 100B model on one CPU at 5 to 7 tok/s, 71.9 to 82.2% less energy on x86 | **Misleading.** "The tested models are dummy setups ... neither trained nor released by Microsoft"; the speedups are against llama.cpp fp16. The 100B figure is 6.58 tok/s on an Apple M2-series machine and 1.70 tok/s on an i7-13700H with 64 GB [C, [arXiv 2410.16144](https://arxiv.org/html/2410.16144)] |
| Binary scoring makes guessing optimal (OpenAI 2025) | True: "the training and evaluation procedures reward guessing over acknowledging uncertainty" [C, [arXiv 2509.04664](https://arxiv.org/abs/2509.04664)]. Already design rule 2 in `agi-asi-survey.md` §5 |
| "A small model which knows no Māori can simply say I don't know" | From OpenAI's blog, not the paper; the paper's text has no "Māori" [S] |
| I-JEPA, competitive in under 72 hours on 16 GPUs | True for a ViT-Huge/14 on ImageNet [C]. It is 1,152 A100-hours for a vision encoder |
| Mouse synapses stay enlarged; erasing them erases the skill, "a clue worth building on" | **Already built on, and it failed.** The passage is the motivation of EWC [V, [Kirkpatrick et al.](https://arxiv.org/abs/1612.00796) introduction]. EWC scores 20.01% against 19.90% with no protection on class-incremental Split-MNIST [C✓, `foundational-survey.md` §4.3] |
| Optimal continual learning needs perfect memory and is NP-hard | True for optimal algorithms [C, Knoblauch et al.]. It does not rule out good-enough learning with bounded memory |
| 4.7 bits per synapse "adds up to at least a petabyte" | Overstated: 59 TB to 0.59 PB (§6.3) |
| Loihi 2 does 30 fps detection at 50 to 100 mW against 15 to 20 W for a Jetson Nano | **No primary source found.** NVIDIA rates the Jetson Nano at "5W - 10W". The nearest Intel paper compares a Jetson Orin Nano on steering regression: "over 150× lower energy", with MSE 0.035 against 0.025 [C, [arXiv 2310.03251](https://arxiv.org/abs/2310.03251)] |
| Hala Point: 1.15 billion neurons, 128 billion synapses, 2,600 W | True [C, Intel] |
| AMI Labs raised a record $1.03B seed at $3.5B | $1.03B true [C]; "record", "seed" and $3.5B are press only [S]. Nothing published [V] |
| Sutton and Javed left Keen to found Oak Lab | The lab is real [C]; the Keen departure is press only [S]. No benchmark result |
| Thousand Brains Project: MIT licence, November 2024; all patents pledged; seven employees | Licence and date true [V]. The pledge covers "the listed patents", and the list returns 404. The team page lists 12 people; the repo lists 7 maintainers [V] |
| VERSES Genius "knows what it doesn't know" | No calibration metric published [V]. VERSES discontinued its AI research on 18 Jun 2026 [C] |
| ARC-AGI-3: "frontier AI scored zero" | True at the July 2025 preview; out of date since. The Kaggle leader is 19.45 (§4) |

---

## 10. Corrections made during this census

Each was caught by the verification pass or by re-measurement.
- **Laptop speed.** The first measurements (122 to 135 GFLOP/s matmul, 6.7 to 7.3 GFLOP/s batch-1) ran under load. Idle: 386 and 14 to 73 (§6.1).
- **Synapse storage.** An early estimate said 88 TB from a neocortex-only count of 1.5e14. The verified count is 1.64e14 for the neocortex, and the whole-brain range gives 59 TB to 0.59 PB.
- **Loss of plasticity.** "89% to 77%" is from the preprint; Nature gives 88% and no 77%.
- **WSCL.** The comparison uses a 200-sample buffer, not 5,000.
- **SRC.** The original paper gives 19.49 to 48.47; "0.20 to 0.53" is a single run in a 2026 follow-up.
- **SIESTA.** Its backbone is not frozen; 97.81% of it trains during sleep.
- **Monty's hierarchy.** First recorded as worse than flat; it is mixed. The run times and hierarchy numbers are from the docs, not the paper.
- **HRM cost.** $148.50 is the ARC-AGI-1 run only.
- **Vicarious.** "A plain CNN" beating RCN overstated it: the winner was a gated fully-convolutional network with augmentation.
- **Stream Q(λ).** The 2026 paper that reports it on 4 CPU cores is a methods paper using it as a baseline, not a replication.
- **Energy figures.** The V100 (0.47 µJ) and SpiNNaker (0.6 µJ) figures come from Knight &amp; Nowotny 2018 and Rhodes et al. 2020, not van Albada 2018. The spiking-brain upper bounds in SoBE 2025 are for Hodgkin-Huxley neurons, not simple spiking neurons.
- **TRM on Kaggle.** An early estimate said 76 weeks, assuming one GPU. At the survey's 2-GPU assumption it is about 38.

## 11. Unverified

- **Hardware and timing.**
  - The CPU behind the fly model's "5 min per ... thread"; "runs on a laptop" is press only [S].
  - Spaun's simulation speed (press only).
  - The laptop's RAM bandwidth on an idle machine. §6.3 uses the provisional 19.8 GB/s.
  - The 24-byte traffic per parameter in §6.3 is an assumption.
- **Scores.**
  - CompressARC's 4% on ARC-AGI-2 appears only in a secondary report.
  - URM and VARC scores are self-reported.
  - Whether the ARC-AGI-3 entries named as LLM-free are the versions currently submitted.
- **Other claims.**
  - A direct whole-brain synapse count; none found.
  - Any independent evaluation of HOPE; none found in 22 of 90 citing papers.
  - The methods behind the 2026 ARC-AGI-3 and ARC-AGI-2 leaders; write-ups are due around 30 Sep 2026.
  - A SoBE 2025 data inconsistency: its whole-brain event-driven upper bound (8.67e16) is below its cortex-only one (1.46e17).

---

## Appendix A. Laptop measurement scripts

### A.1 Compute benchmark (§6.1)

Run on 2026-09-26 with load average 0.35 beforehand, Python 3.12.3, numpy 2.5.1. The script below was run inline, once per thread count:

```bash
for t in 1 4 8; do OPENBLAS_NUM_THREADS=$t python3 - <<'EOF' ... EOF; done
```

```python
import os
import time

import numpy as np

n = 4096
rng = np.random.default_rng(0)
a = rng.standard_normal((n, n), dtype=np.float32)
b = rng.standard_normal((n, n), dtype=np.float32)
a @ b
ts = []
for _ in range(5):
    t = time.perf_counter()
    a @ b
    ts.append(time.perf_counter() - t)
flop = 2 * n**3
print(
    f"threads={os.environ['OPENBLAS_NUM_THREADS']} fp32 matmul n={n} GFLOP/s per run="
    f"{[round(flop / t / 1e9, 1) for t in ts]} best={flop / min(ts) / 1e9:.1f}",
    flush=True,
)
for d in (512, 1024, 2048):
    w = rng.standard_normal((d, d), dtype=np.float32)
    x = rng.standard_normal(d, dtype=np.float32)
    w @ x
    reps = 2000
    t = time.perf_counter()
    for _ in range(reps):
        w @ x
    dt = (time.perf_counter() - t) / reps
    print(f"  batch-1 layer d={d}: {2 * d * d / dt / 1e9:.2f} GFLOP/s, {dt * 1e6:.1f} us per layer", flush=True)
```

Output:

```
threads=1 fp32 matmul n=4096 GFLOP/s per run=[107.7, 106.5, 115.0, 123.8, 109.7] best=123.8
  batch-1 layer d=512: 36.70 GFLOP/s, 14.3 us per layer
  batch-1 layer d=1024: 24.41 GFLOP/s, 85.9 us per layer
  batch-1 layer d=2048: 13.93 GFLOP/s, 602.1 us per layer
threads=4 fp32 matmul n=4096 GFLOP/s per run=[386.1, 264.5, 355.7, 344.2, 273.4] best=386.1
  batch-1 layer d=512: 31.56 GFLOP/s, 16.6 us per layer
  batch-1 layer d=1024: 72.11 GFLOP/s, 29.1 us per layer
  batch-1 layer d=2048: 25.55 GFLOP/s, 328.3 us per layer
threads=8 fp32 matmul n=4096 GFLOP/s per run=[359.6, 337.7, 352.4, 303.8, 350.3] best=359.6
  batch-1 layer d=512: 28.73 GFLOP/s, 18.3 us per layer
  batch-1 layer d=1024: 73.17 GFLOP/s, 28.7 us per layer
  batch-1 layer d=2048: 18.33 GFLOP/s, 457.6 us per layer
```

### A.2 ARC systems on the CPU (§4)

Setup, torch 2.14.0+cpu, numpy 2.5.3:

```bash
uv venv venv --python 3.12
VIRTUAL_ENV=$PWD/venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch
VIRTUAL_ENV=$PWD/venv uv pip install numpy matplotlib scipy einops pydantic omegaconf hydra-core huggingface_hub tqdm coolname argdantic
git clone https://github.com/iliao2345/CompressARC.git                      # HEAD 83a2221
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git    # HEAD c011037
git clone https://github.com/lillian039/VARC.git                            # commit not recorded
git clone https://github.com/DriesSmit/ARC3-solution.git                    # commit not recorded
git clone https://github.com/top-quarks/ARC-solution.git icecuber           # commit not recorded
curl -sSL -o trm_ckpt/step_518071 "https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/step_518071"
```

Each timing below is one run over a short window (3 to 20 steps). The scripts are recorded exactly as they ran, so they are fenced as plain text and kept out of the formatter.

**Matmul ceiling in that session** (`venv/bin/python - <<'EOF' ... EOF`):

```text
import time, torch
for th in (4, 8):
    torch.set_num_threads(th)
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n); b = torch.randn(n, n)
        a @ b
        t = time.time(); r = 5
        for _ in range(r): a @ b
        dt = (time.time() - t) / r
        print(f'threads={th} n={n} fp32 matmul {2*n**3/dt/1e9:.0f} GFLOPS', flush=True)
a = torch.randn(2048, 2048, dtype=torch.bfloat16); b = torch.randn(2048, 2048, dtype=torch.bfloat16)
torch.set_num_threads(4); a @ b
t = time.time()
for _ in range(5): a @ b
print(f'threads=4 n=2048 bf16 matmul {2*2048**3/((time.time()-t)/5)/1e9:.0f} GFLOPS', flush=True)
```

```
threads=4 n=1024 fp32 matmul 128 GFLOPS
threads=4 n=2048 fp32 matmul 132 GFLOPS
threads=4 n=4096 fp32 matmul 133 GFLOPS
threads=8 n=1024 fp32 matmul 121 GFLOPS
threads=8 n=2048 fp32 matmul 131 GFLOPS
threads=8 n=4096 fp32 matmul 133 GFLOPS
threads=4 n=2048 bf16 matmul 40 GFLOPS
```

**CompressARC step time**: `venv/bin/python time_compressarc_cpu.py evaluation 16 20 4`. The script is below.

```text
import json
import sys
import time

import numpy as np
import torch

_real_set_default_device = torch.set_default_device
torch.set_default_device = lambda d: _real_set_default_device('cpu')

sys.path.insert(0, 'CompressARC')
import preprocessing
import arc_compressor
import train
import solution_selection

split = sys.argv[1]
n_tasks = int(sys.argv[2])
n_steps = int(sys.argv[3])
threads = int(sys.argv[4])
torch.set_num_threads(threads)
print(f'torch {torch.__version__} threads={torch.get_num_threads()} split={split} n_tasks={n_tasks} n_steps={n_steps}', flush=True)

with open(f'CompressARC/dataset/arc-agi_{split}_challenges.json') as f:
    problems = json.load(f)
names = sorted(problems)
rng = np.random.default_rng(0)
picked = [names[i] for i in rng.choice(len(names), n_tasks, replace=False)]
if len(sys.argv) > 5:
    picked = sys.argv[5].split(',')

rows = []
for name in picked:
    torch.manual_seed(0)
    task = preprocessing.Task(name, problems[name], None)
    model = arc_compressor.ARCCompressor(task)
    n_params = sum(p.numel() for p in model.weights_list)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    logger = solution_selection.Logger(task)
    logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for _ in range(task.n_test))
    logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for _ in range(task.n_test))
    train.take_step(task, model, optimizer, 0, logger)
    t0 = time.time()
    for step in range(1, n_steps + 1):
        train.take_step(task, model, optimizer, step, logger)
    dt = (time.time() - t0) / n_steps
    max_hw = max(max(s[0][0], s[0][1], s[1][0], s[1][1]) for s in task.shapes if s[1] is not None) if task.shapes else 0
    rows.append((name, task.n_examples, max_hw, n_params, dt))
    print(f'{name} n_examples={task.n_examples} n_x={task.n_x} n_y={task.n_y} n_colors={task.n_colors} params={n_params} sec/step={dt:.3f} -> 2000 steps = {dt*2000/60:.1f} min', flush=True)

dts = np.array([r[4] for r in rows])
print(f'mean sec/step={dts.mean():.3f} median={np.median(dts):.3f} min={dts.min():.3f} max={dts.max():.3f}', flush=True)
print(f'projected 2000 steps/task: mean {dts.mean()*2000/60:.1f} min; 400 tasks: {dts.mean()*2000*400/3600:.0f} h', flush=True)
```

The `params=` column counts every tensor in `weights_list`, including per-puzzle latents, so it is not the network size.
- **Run 1:** 16 tasks. The first five overlapped another job.
- **Run 2:** re-ran those five alone with `... 4ff4c9da,9f27f097,845d6e51,0becf7df,bf89d739`. Their times were 1.758, 1.187, 1.717, 1.146 and 1.261 s/step.
- **Combined:** Run 1's other 11 values plus Run 2's five give min 1.046, median 1.253, mean 1.404 and max 2.539 s/step. At 2000 steps that is a median of 41.8 min per task, and 94 h for 120 tasks.
- **Missing name:** one of Run 1's 16 lines (1.046 s/step) was printed but its task name was not kept.

Run 1 output as kept:

```
torch 2.14.0+cpu threads=4 split=evaluation n_tasks=16 n_steps=20
4ff4c9da n_examples=4 n_x=27 n_y=27 n_colors=3 params=1741860 sec/step=1.841 -> 2000 steps = 61.4 min
9f27f097 n_examples=4 n_x=12 n_y=12 n_colors=5 params=1130868 sec/step=1.143 -> 2000 steps = 38.1 min
845d6e51 n_examples=4 n_x=18 n_y=17 n_colors=7 params=1626948 sec/step=1.651 -> 2000 steps = 55.0 min
0becf7df n_examples=4 n_x=10 n_y=10 n_colors=8 params=1152684 sec/step=2.791 -> 2000 steps = 93.0 min
bf89d739 n_examples=5 n_x=19 n_y=17 n_colors=2 params=1227412 sec/step=3.050 -> 2000 steps = 101.7 min
9c56f360 n_examples=4 n_x=9 n_y=9 n_colors=2 params=925020 sec/step=1.140 -> 2000 steps = 38.0 min
d2acf2cb n_examples=4 n_x=10 n_y=9 n_colors=4 params=997164 sec/step=1.149 -> 2000 steps = 38.3 min
e7b06bea n_examples=6 n_x=14 n_y=9 n_colors=9 params=1487108 sec/step=1.537 -> 2000 steps = 51.2 min
d56f2372 n_examples=4 n_x=22 n_y=20 n_colors=7 params=1951812 sec/step=2.539 -> 2000 steps = 84.6 min
140c817e n_examples=4 n_x=14 n_y=14 n_colors=6 params=1292508 sec/step=1.213 -> 2000 steps = 40.4 min
f823c43c n_examples=3 n_x=16 n_y=19 n_colors=6 params=1352996 sec/step=1.353 -> 2000 steps = 45.1 min
45737921 n_examples=4 n_x=13 n_y=12 n_colors=7 params=1258308 sec/step=1.201 -> 2000 steps = 40.0 min
84f2aca1 n_examples=5 n_x=16 n_y=13 n_colors=6 params=1438660 sec/step=1.462 -> 2000 steps = 48.7 min
0692e18c n_examples=4 n_x=9 n_y=9 n_colors=4 params=982764 sec/step=1.246 -> 2000 steps = 41.5 min
281123b4 n_examples=7 n_x=4 n_y=19 n_colors=5 params=1141212 sec/step=1.507 -> 2000 steps = 50.2 min
mean sec/step=1.617 median=1.408 min=1.046 max=3.050
projected 2000 steps/task: mean 53.9 min; 400 tasks: 359 h
```

**TRM checkpoint shapes** (`venv/bin/python - <<'EOF' ... EOF`):

```text
import torch
sd = torch.load('trm_ckpt/step_518071', map_location='cpu', weights_only=False)
print(type(sd), len(sd))
tot = 0; emb = 0
for k, v in sd.items():
    if hasattr(v, 'shape'):
        n = v.numel(); tot += n
        if 'puzzle_emb' in k: emb += n
        if n > 100000 or 'puzzle' in k:
            print(k, tuple(v.shape), v.dtype)
print(f'total params {tot:,}; puzzle_emb {emb:,}; trunk {tot-emb:,}')
```

```
<class 'collections.OrderedDict'> 15
_orig_mod.model.inner.puzzle_emb.weights (876406, 512) torch.float32
_orig_mod.model.inner.L_level.layers.0.self_attn.qkv_proj.weight (1536, 512) torch.float32
_orig_mod.model.inner.L_level.layers.0.self_attn.o_proj.weight (512, 512) torch.float32
_orig_mod.model.inner.L_level.layers.0.mlp.gate_up_proj.weight (3072, 512) torch.float32
_orig_mod.model.inner.L_level.layers.0.mlp.down_proj.weight (512, 1536) torch.float32
_orig_mod.model.inner.L_level.layers.1.self_attn.qkv_proj.weight (1536, 512) torch.float32
_orig_mod.model.inner.L_level.layers.1.self_attn.o_proj.weight (512, 512) torch.float32
_orig_mod.model.inner.L_level.layers.1.mlp.gate_up_proj.weight (3072, 512) torch.float32
_orig_mod.model.inner.L_level.layers.1.mlp.down_proj.weight (512, 1536) torch.float32
total params 455,549,954; puzzle_emb 448,719,872; trunk 6,830,082
```

**TRM inference time**: `venv/bin/python time_trm_cpu.py 8`, with random inputs, fp32 and the checkpoint's own configuration.

```text
import sys, time
import torch
sys.path.insert(0, 'TinyRecursiveModels')
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

torch.set_num_threads(4)
B = int(sys.argv[1])
cfg = dict(batch_size=B, seq_len=900, puzzle_emb_ndim=512, num_puzzle_identifiers=876406, vocab_size=12,
           H_cycles=3, L_cycles=4, H_layers=0, L_layers=2, hidden_size=512, expansion=4, num_heads=8,
           pos_encodings='rope', halt_max_steps=16, halt_exploration_prob=0.1, forward_dtype='float32',
           mlp_t=False, puzzle_emb_len=16, no_ACT_continue=True)
model = TinyRecursiveReasoningModel_ACTV1(cfg)
sd = torch.load('trm_ckpt/step_518071', map_location='cpu', weights_only=False)
sd = {k.replace('_orig_mod.model.', ''): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing', missing, 'unexpected', unexpected, flush=True)
model.eval()
batch = {'inputs': torch.randint(0, 12, (B, 900)), 'labels': torch.randint(0, 12, (B, 900)), 'puzzle_identifiers': torch.randint(1, 876406, (B,))}
with torch.no_grad():
    carry = model.initial_carry(batch)
    carry, out = model(carry, batch)
    t = time.time()
    for _ in range(3):
        carry, out = model(carry, batch)
    per_act = (time.time() - t) / 3
per_sample_full = per_act * 16 / B
print(f'batch={B}: {per_act:.2f} s per ACT step (15 network applications); full 16-step eval = {per_sample_full:.2f} s per augmented sample', flush=True)
print(f'1000 augmentations for one test input = {per_sample_full*1000/3600:.2f} h; ARC-AGI-1 eval (419 test inputs) x 1000 aug = {per_sample_full*1000*419/3600/24:.1f} days', flush=True)
```

```
missing [] unexpected []
batch=8: 19.44 s per ACT step (15 network applications); full 16-step eval = 38.88 s per augmented sample
1000 augmentations for one test input = 10.80 h; ARC-AGI-1 eval (419 test inputs) x 1000 aug = 188.6 days
```

**VARC train step**: `venv/bin/python time_varc_cpu.py`.
- timm's `PatchEmbed` is replaced by an equivalent strided convolution, because timm failed to import on this torch build.
- AdamW with weight decay 0 stands in for the paper's Adam.
- The script's "paper uses 2 runs" label is wrong: the 2 runs come from the repo's test-time training script, and the paper's ensemble used four.

```text
import sys, time
import torch
sys.path.insert(0, 'VARC/src'); sys.path.insert(0, 'VARC')
import types
import torch.nn as nn
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
_m = types.ModuleType('timm.models.vision_transformer'); _m.PatchEmbed = PatchEmbed
sys.modules['timm'] = types.ModuleType('timm'); sys.modules['timm.models'] = types.ModuleType('timm.models'); sys.modules['timm.models.vision_transformer'] = _m
from ARC_ViT import ARCViT

torch.set_num_threads(4)
torch.manual_seed(0)
model = ARCViT(num_tasks=51, image_size=64, num_colors=12, embed_dim=512, depth=10, num_heads=8, mlp_dim=512, patch_size=2)
n = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
x = torch.randint(0, 12, (8, 64, 64))
mask = torch.zeros(8, 64, 64, dtype=torch.bool); mask[:, :30, :30] = True
y = torch.randint(0, 12, (8, 64, 64))
t_ids = torch.randint(0, 51, (8,))
model.train()
def step():
    logits = model(x, t_ids, attention_mask=mask)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward(); opt.step(); opt.zero_grad()
step()
t = time.time()
for _ in range(5):
    step()
dt = (time.time() - t) / 5
model.eval()
with torch.no_grad():
    xb = torch.randint(0, 12, (30, 64, 64)); mb = torch.ones(30, 64, 64, dtype=torch.bool); tb = torch.zeros(30, dtype=torch.long)
    model(xb, tb, attention_mask=mb)
    t = time.time(); model(xb, tb, attention_mask=mb); fwd30 = time.time() - t
steps_per_ttt = 100 * ((51 * 3 + 7) // 8)
print(f'params={n} train_step(batch8)={dt:.2f}s  steps per TTT run (100 epochs x 51 aux tasks x 3 pairs / batch 8)={steps_per_ttt}', flush=True)
print(f'one TTT run = {dt*steps_per_ttt/60:.1f} min; paper uses 2 runs per task (ttt-num-each 2) = {2*dt*steps_per_ttt/60:.1f} min', flush=True)
print(f'inference fwd 30 views={fwd30:.2f}s -> 510 views = {fwd30*17:.1f}s', flush=True)
```

```
Patch size: 2, sequence length: 1024
params=17411120 train_step(batch8)=34.22s  steps per TTT run (100 epochs x 51 aux tasks x 3 pairs / batch 8)=2000
one TTT run = 1140.6 min; paper uses 2 runs per task (ttt-num-each 2) = 2281.2 min
inference fwd 30 views=40.95s -> 510 views = 696.1s
```

**StochasticGoose CNN**: `venv/bin/python time_goose_cpu.py`. `out.sum()` stands in for the repo's loss.

```text
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

src = open('ARC3-solution/custom_agents/action.py').read()
start = src.index('class ActionModel(nn.Module):')
end = src.index('class Action(Agent):')
exec(src[start:end])

for threads in (4, 8):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    m = ActionModel(input_channels=16, grid_size=64)
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    x1 = torch.rand(1, 16, 64, 64)
    xb = torch.rand(64, 16, 64, 64)
    with torch.no_grad():
        m(x1)
        t = time.time()
        for _ in range(20):
            m(x1)
        fwd1 = (time.time() - t) / 20
    m(xb).sum().backward(); opt.step(); opt.zero_grad()
    t = time.time()
    for _ in range(3):
        out = m(xb)
        out.sum().backward()
        opt.step(); opt.zero_grad()
    step = (time.time() - t) / 3
    print(f'threads={threads} params={n} fwd(batch1)={fwd1*1000:.1f} ms train_step(batch64)={step:.2f} s', flush=True)
```

Clean run. An earlier run overlapped another job and is not used.

```
threads=4 params=34320614 fwd(batch1)=43.8 ms train_step(batch64)=8.76 s
threads=8 params=34320614 fwd(batch1)=35.4 ms train_step(batch64)=8.20 s
```

The per-action cost is 0.0438 + 8.76 / 5 = 1.79 s, because the agent trains every 5 actions (`action.py` lines 149 to 153).

**icecuber, depth 2**: the repo's own `run.py`, unmodified, then `summary.py`.
- Wall time: `real 2m41.866s`, `user 13m35.260s`.
- `summary.py` printed `Total: 419`, `Size : 404`, `Cands: 138`, `Correct: 129`, and 0.1 GB peak memory per worker.
- The per-input result string is omitted here.
- Re-scoring the candidates by rank gave top-1 112/419, top-2 120/419 (28.6%) and top-3 129/419.
