# Modeling Team — One Pager

A modeling team that compresses capability from strong, expensive models into small models that are cheap and fast to serve. The work is defined by two constraints: **governance-safe** and **efficient**.

## Goal

Take a strong teacher model and produce a small model that hits a target size, latency, and serving cost while retaining as much quality as possible. Serving cost dominates the lifetime cost of a deployed model — trained once, served indefinitely — so reducing it is the primary return.

- **Efficient** — the teacher is a swappable input, not a dependency. Use the strongest available model, re-distill when a better one appears. The params a requesting team sets are mostly how small, fast, and cheap the served model must be.
- **Governance-safe** — the model we ship must be built on an approved, policy-safe base. Many of the strongest models can't be deployed for policy/licensing reasons, but they're fine to use as *teachers*. Distillation is precisely what bridges this: take capability from any strong teacher and put it into an approved student we're allowed to ship. The deployed artifact is clean because the base is on the allowed list, even though its capability came from a model that isn't. Safety/red-team gates still apply before a model is marked shippable (especially guardrail models).

## Budget

Primary cost drivers (to be sized against target volume):

| Line | Driver |
|---|---|
| Teacher inference | Data generation + scoring signal (largest variable cost; external API teachers add per-token cost) |
| Training + rollout compute | SFT, offline KD, and on-policy/rollout loops (rollouts are the most compute-heavy) |
| Serving | The cost we are reducing; distilled models cut it structurally |
| Headcount | 4 (Lead + 3), see Assignment |
| Tooling | Experiment tracking, data store, eval harness, sandbox for tool-use training |

Note: training compute is a one-time cost per model; serving savings recur. The budget trade is "spend training compute once to lower serving cost permanently."

## Assignment

| Role | Owns |
|---|---|
| Data & curation | Data generation + metrics-based filtering |
| Distillation researcher | Offline + on-policy distillation; scoring-signal design (verifiers/judges are a real sub-problem, not a drop-in) |
| Training & inference optimization | Efficient training (QAT, mixed precision, distillation training loop), inference optimization (quantization, pruning, latency/throughput tuning), and serving-cost reduction |
| Lead | Job-spec schema, governance rules, cross-modality strategy, hands-on where the bottleneck is |

Audio is a phased expansion, not a separate headcount — ideally one hire has speech background.

## Tech Development

**Scope** — three areas:

- **Training data** — data/metric generation (synthetic data from teacher models, across text and audio), metrics-based filtering (quality/difficulty scoring, dedup, decontamination), and a data flywheel that recycles model outputs and eval signals back into the next round of training data.
- **Training** — knowledge distillation (SFT + offline from teacher traces), on-policy distillation (learning from a signal on the student's own generations), thought/reasoning-process distillation, and tool-orchestration distillation (multi-step tool-use trajectories).
- **SLM** — quantization-aware training (QAT), pruning, and distill-to-small; kept inside training because compression and quality-recovery share the same teacher, data, and eval loop. QAT fits naturally: the teacher signal recovers the quality lost to lower precision during training rather than after.

Across all training methods the student learns from a signal on its own generations, anchored to the teacher for stability. The signal is a parameter:
- **teacher logits / distribution** — reproduce teacher behavior from the student's own states
- **judge** — open-ended tasks with no ground truth
- **verifier / environment** — checkable tasks (math, code, tool calls); here the student can beat the teacher because the signal is ground truth, not imitation

Reasoning-process and tool-use capabilities follow the same path: filtered teacher traces → on-policy correction → verifiable signal pushing past the teacher. The signal source is a config field, so sophistication scales without changing the method.

## Roadmap

1. **Foundations (done)** — text and audio TTS distilled end-to-end with offline KD+SFT and reproducible eval.
2. **Improve methods** — add on-policy distillation and a verifiable-signal path on top of offline KD+SFT; strengthen filtering and eval gates.
3. **Parameterize** — a single job spec drives the whole pipeline; extend to ASR.
4. **Templates + gates** — use-case templates (guardrail LLM, domain SLM) with automated quality and safety acceptance gates.
5. **Self-serve (limited)** — partner teams submit jobs themselves; tenant isolation, quotas, cost attribution, model registry, API access controls.
6. **Harden + govern + extend** — enforce approved student-base list, safety/red-team gates, and shippability checks; open to broader internal use and add further audio tasks as templates.

Principle throughout: productize a pipeline only after it works by hand. The job spec and templates are the product; isolation, quotas, and governance are what make it safe to hand to other teams.

## Current State

Distillation is working on **text** and **audio TTS**, using **offline knowledge distillation + SFT only**. Two results already in hand:
- **Text — guardrailing**: distilled a large model's safety judgments into a small LLM; the lead candidate for the first self-serve use case.
- **Audio — TTS**: shipped a governance-safe TTS model to production.

Two improvement axes from here:
- **Methods** — move beyond offline KD+SFT to on-policy distillation and verifiable-signal training, so students can match and (on checkable tasks) exceed the teacher rather than only imitate it.
- **Modalities** — extend audio beyond TTS to **ASR** and further speech tasks.

## Milestones

**3 months — deepen methods on existing modalities**
- Add on-policy distillation on top of offline KD+SFT for text and TTS, fixing the distribution drift that offline-only training leaves
- Improve metrics-based filtering; stand up eval gates and a governance baseline (approved student-base list, safety gate before shippable)
- Start ASR data generation and pipeline work

**6 months — extend to ASR + verifiable signals**
- ASR distillation working end-to-end to a WER target, running alongside TTS
- Verifiable-signal path live (verifier/environment scoring) for checkable tasks, with students beating the teacher where ground truth exists
- SLM optimization (quantization-aware training, pruning) folded in for concrete serving-cost cuts
- Use-case templates + first partner team self-serving via job spec

**12 months — platform + scale**
- Full method stack selectable per job: offline KD → on-policy → verifiable signal
- Modalities as templates: text, TTS, ASR (and the next audio task)
- Broader internal self-serve with full governance (provenance/PII/license scanning, audit trail, red-team gates)
- Multiple production models deployed with tracked serving-cost savings across teams
