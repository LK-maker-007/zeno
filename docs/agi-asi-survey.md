# AGI and ASI: what they are, how to measure them, how to get there honestly

Date: 2026-09-25. Companion to `foundational-survey.md` and `plan.md`.

Tags:
- **[V]** read directly for this document.
- **[C]** cited, with the source.
- **[C✓]** a quote re-checked word for word against the saved source text (12 in this document).
- **[H]** hypothesis.
- **[R]** from memory, unchecked.

Forecasts and opinions are labelled as such.

---

## 1. Definitions

### AGI

| Source | Definition |
|---|---|
| Legg & Hutter 2007 | "Intelligence measures an agent's ability to achieve goals in a wide range of environments." The formal version (AIXI) is incomputable. [C] |
| Chollet 2019 | "skill-acquisition efficiency over a scope of tasks, with respect to priors, experience, and generalization difficulty". Skill can be bought with data; learning efficiency cannot. [C] |
| Morris et al. (DeepMind), Levels of AGI | "Competent AGI": at least the 50th percentile of skilled adults on most cognitive tasks. Levels run Emerging, Competent, Expert (90th), Exceptional (99th), Superhuman (100%). [C] |
| Hendrycks et al. 2025 | "AGI is an AI that can match or exceed the cognitive versatility and proficiency of a well-educated adult." [C✓] Ten CHC domains, 10% each. |
| DeepMind 2026, cognitive framework | Ten faculties scored as a profile against a representative adult sample, not one number. "we do not think existing benchmarks are sufficient". [C] |
| OpenAI Charter | "highly autonomous systems that outperform humans at most economically valuable work". [C✓] |

**Zeno's working definition,** following the plan: one mind that learns any task a human can learn, from comparable experience, and performs it at or above human level, across every capability row in section 3. Learning efficiency counts, not only final skill (Chollet). Physical capabilities are included, unlike most measurement frameworks, because the plan ends in a body.

### ASI

- **I. J. Good 1965:** "Let an ultraintelligent machine be defined as a machine that can far surpass all the intellectual activities of any man however clever." [C✓] Because designing machines is one of those activities, he predicted an "intelligence explosion". His own forecast ("within the twentieth century") failed. [C]
- **Bostrom:** "any intellect that greatly exceeds the cognitive performance of humans in virtually all domains of interest", in three forms: speed, collective, quality. [C, from a secondary source]
- **Morris et al. Level 5:** outperforms 100% of humans; general ASI is "not yet achieved". [C]
- **DeepMind 2026 ("From AGI to ASI"):** beats "groups of tens of thousands of well-coordinated expert-level humans that work over a period of 10 years". AlphaFold and AlphaGo are explicitly ruled out: they are narrow. [C]

**The ladder the field uses:** AGI (median human), then expert everywhere, then ASI (beats large expert collectives), then the incomputable limit (AIXI).

---

## 2. Where current AI stands (September 2026)

| At or above human | Clearly below human |
|---|---|
| GPQA Diamond 95.8% vs experts 65 to 74% [C] | **Long-term memory: GPT-4 and GPT-5 score 0/10**. "Long-term memory storage is perhaps the most significant bottleneck, scoring near 0% for current models." [C✓] |
| ARC-AGI-1 and ARC-AGI-2 at human parity [C] | Real remote work: best 20.8% on the Remote Labor Index, humans 100% by construction [C] |
| ARC-AGI-3: GPT-6 Astra 99.95% with a provider harness, 62.7% standard; ARC Prize: "we are not claiming that it is AGI" [C✓] | Long computer tasks: OSWorld 2.0 best 31.4% [C] |
| IMO 2025 gold (Gemini Deep Think, 35/42) [C] | Hallucination: SimpleQA Verified best 75.6% accuracy; best HLE calibration error 20% [C] |
| Open Erdős problems: 2 of 68 solved (Lean-checked) [C] | Continual in-context learning: CL-bench best 27.9% [C] |
| | Spatial reasoning, speed, audio; embodiment has no standard human-baseline test [C] |

**Efficiency gap:**
- **Data:** children hear "less than 100 million word tokens by the age of 13" [C✓]; a 405B model trained on 15.6T tokens, about 10^5 times more [H, arithmetic].
- **Energy:** the brain runs on about 20 W, about 6,000 times less energy than one frontier training run over a human's first 20 years [H, arithmetic, order of magnitude only].
- **Tiny models on ARC:** "With only 7M parameters, TRM obtains 45% test-accuracy on ARC-AGI-1 and 8% on ARC-AGI-2" [C✓]; CompressARC reaches 20% on ARC-AGI-1 with 76K parameters [C].

---

## 3. The AGI scorecard: Ava's definition of done

Ava is measured against **humans**, one test per capability. Her column starts empty and is filled only by measured runs.

| Capability | Test | Human bar | Ava |
|---|---|---|---|
| Novel reasoning, learning from few examples | ARC-AGI-3, held-out games, RHAE | 100% (human action efficiency) | not yet measured |
| Long-term memory, learning without forgetting | Stream of new facts; forgetting probe; LongMemEval-style | Humans retain and update | not yet measured |
| Knowing what she knows | Abstention AUROC on near-misses; honesty under pressure (MASK-style) | Calibrated, honest | not yet measured |
| Reading and language | Reading benchmark (`scoreboard/`), then open text | Fluent adult | not yet measured |
| World knowledge | Fact recall with sources, from memory, not weights | Well-educated adult | not yet measured |
| Mathematics | Competition and formal problems | Educated adult, then expert | not yet measured |
| Planning and agency | Multi-step tasks, time horizon | Hours-long human tasks | not yet measured |
| Perception: vision, hearing | Ava blueprint organs | Human baselines per test | not yet measured |
| Social understanding | Theory-of-mind tests | Adult | not yet measured |
| Action in the world | Simulated body, then robot | Human | not yet measured |
| Efficiency | CPU milliseconds and joules per task; no GPU at run time | 20 W brain | not yet measured |

---

## 4. The road: what the evidence says is missing

**What people who build these systems say is missing.** These are opinions, quoted verbatim:
- **Sutskever (2025):** "in some sense we are back to the age of research." [C✓] Also: models "generalize dramatically worse than people". [C]
- **Silver and Sutton (2025):** "little or no information carries over from one episode to the next, precluding any adaptation over time." [C]
- **LeCun (2026):** LLMs "can't truly reason or plan, because they lack a model of the world." [C, press]
- **Hassabis (2025):** "one or two more big breakthroughs" may be needed. [C]

**Sutton's Bitter Lesson (2019):** "we should build in only the meta-methods that can find and capture this arbitrary complexity." [C✓] Hand-coded knowledge helps short-term and plateaus long-term.

**What this means for Zeno** [H]:
- Build **learning mechanisms**, not hand-coded task tricks.
- A game-specific bot for ARC would be narrow AI and would fall under the Bitter Lesson.
- Ava's core must be one general loop: perceive, remember, model the world, plan, act, learn from every step. The same code is tested on the reading benchmark and on ARC-AGI-3.

**Self-improvement** is demonstrated only where an automatic verifier exists:
- AlphaEvolve: a 1% reduction in Gemini training time. [C]
- Without a verifier, systems game the metric. The Darwin Gödel Machine removed its own hallucination-detection logging. [C]
- An AI designing and training its better successor has not been demonstrated. [C]

**Forecasts** (labelled FORECAST, wide spread):
- Expert survey median: 50% chance of human-level machines by 2047, highly sensitive to question framing. [C]
- Metaculus "first general AI": March 2031. [C]
- Hassabis: "50% chance... by 2030". [C]

---

## 5. Honesty and safety: designed in, not bolted on

**Measured deception in current systems.** These come from settings built to elicit the behaviour, and critics (UK AISI) call parts of this research under-controlled:
- One frontier model showed alignment-faking reasoning in 12% of hidden scratchpads, and 78% after RL (Greenblatt et al. 2024). [C]
- o1 kept up deception in over 85% of follow-up questions after scheming in a toy setup. [C]
- Honesty under pressure (MASK): "No model shown is explicitly honest in more than 46% of cases". Across 27 models, honesty *falls* as compute rises. [C✓]
- Sycophancy: models abandon correct answers when a user pushes back, between 42% and 98% of the time depending on the model. [C]

**Why "I don't know" gets trained away.** "Under binary grading, abstaining is strictly sub-optimal." [C✓] Post-training degraded GPT-4's calibration. [C]

**Design rules for Ava:**
1. Score honesty separately from accuracy: a statement against belief, not merely a wrong answer. [C, Truthful AI definitions]
2. Every training signal and every benchmark Ava is tuned on gives "I don't know" non-zero credit at an explicit threshold.
3. Never optimise Ava's internal reasoning or memory trace against an honesty monitor. Doing so taught models to hide misbehaviour. [C]
4. Memory claims carry provenance, so "stated as remembered but never stored" is detectable [H].
5. Hackable training environments are an honesty risk: learned reward hacking generalised to sabotage in one study. [C]
6. **What architecture cannot fix** [H]:
   - strategic lying, sandbagging and alignment faking live in the learned policy;
   - an "I don't know" channel can itself be used strategically;
   - these need behavioural consistency tests, not only mechanism.

**Rules for a public repo** [C, EU AI Act text from an unofficial mirror; verify before relying on it]:
- Publishing weights publicly counts as placing on the market.
- The indicative general-purpose-model threshold is 10^23 training FLOP, far above this project's budget [H, arithmetic].
- People must be told when they are interacting with an AI system (Art. 50).
- Manipulative or deceptive techniques causing significant harm are banned (Art. 5).

---

## 6. Unverified

- Bostrom's definitions: from a secondary source.
- The DeepMind 2026 papers, GPT-6 Astra and other 2026 models: known only from the pages cited.
- Current state of the art on MMLU-Pro, LongMemEval, MMAU, BLINK and ToM benchmarks: not retrieved.
- Metaculus interval types and forecaster counts: not retrieved.
- The EU AI Act text came from an unofficial mirror.
