# WM Visualizer — interpretability tooling and feature-importance findings for an IRIS world model

*A synthesis report. The detailed running log is [WRITEUP.md](WRITEUP.md); the narrative arc is
[STORY.md](STORY.md); the transferable, project-independent lessons are [FINDINGS.md](FINDINGS.md).
This document ties them together and is honest about both the results and the process.*

## Abstract

This body of work rebuilt the WM Visualizer — an interactive tool for inspecting and steering
SAE features in the residual stream of IRIS, a transformer world model trained on Atari
Breakout — and used it to investigate one question: **what does "feature importance" mean for a
world-model SAE, and do the features correspond to the on-screen events a human expects?** Two
results came out of it. First, an apparent "decoupling" between features and game events was
diagnosed to two real causes — a one-frame display bug and, more importantly, the fact that
**activation-magnitude ranking surfaces persistent features (a ball-tracker) rather than event
features (collision detectors), which exist and are robust but rank lower**. Second, an attempt
to build a *causal* importance ranking produced a **clean negative result**: done naively it
collapses to a magnitude ranking (a confound), and done correctly it is too single-state-fragile
to be a trustworthy ranking on this substrate. Of three proposed importance axes, two shipped
(magnitude, temporal stability); causal was demoted to a characterization tool. The negative
result, and *why* it happened, is the more transferable output than a third ranking would have
been.

## 1. What was built

The tool was redesigned from a v1 panel grid into a **canvas of pinned feature cards** (the v2
frontend, `frontend/`; v1 preserved at `frontend-v1/`), importing patterns from LLM-SAE tools
(Neuronpedia, EffectVis): features have stable identity, search-first discovery, persistent
pinning. Concretely shipped:

- A **react-konva canvas** of draggable feature cards (sparkline, intervention-scale slider,
  editable label), backed by a server-side pinned-feature store so layout survives reloads.
- **Search** by feature id or label; a **discovery panel** of candidate features to pin.
- **Multi-feature intervention**: the rollout endpoint now sums several features' directions, so
  the canvas's non-zero-scale cards jointly steer a paused, N-step imagined rollout; baseline vs
  intervened frames + divergence/trajectory charts beneath.
- An **autointerp pipeline** (built, not run by default): top-activating frames per feature →
  vision LLM → cached label, served by a `/feature/{id}` endpoint.
- A **two-axis discovery ranking** — see below.

All of this sits on the unchanged IRIS hook/SAE/imagination infrastructure; the redesign is a
tool rewrite, not a model change.

## 2. The research question: importance on a world-model substrate

The LLM-SAE convention ranks features by **activation magnitude** on the current token. On a
world model this transfers badly: frames are highly correlated, state changes slowly, and
"top-firing right now" churns under the user and is dominated by tail noise. The project proposed
three substrate-adapted alternatives and tested them:

- **Temporal stability** (shipped) — rank by *consistency* of firing over a window. The naive
  version (lowest variance) is degenerate: dead features have variance 0 and look "perfectly
  stable," so it needs a firing-rate gate and a scale-invariant coefficient of variation.
- **Causal importance** (investigated, demoted) — rank by how much a feature's intervention
  changes an imagined rollout. This is §4.
- (Magnitude is kept as the third toggle, explicitly as the "churny LLM convention," for
  contrast.)

## 3. Headline result: the feature–event "decoupling," explained

Using the tool surfaced a real complaint: SAE features looked *decoupled* from on-screen events —
firing while the ball was mid-air, flat during paddle-ball collisions. A five-test diagnostic
battery (Tests 1–3, 5; `backend/scripts/diagnostics/`) resolved it. The features are **not
broken** — two distinct, real causes were at work:

1. **A one-frame display offset (fixed).** The displayed game frame was captured *after*
   `env.step()` while the activations beside it describe the *pre-step* obs — so the frame led its
   own activations by one step. Measured as a clean +1 lag, reproducible across episodes; fixed by
   capturing the frame before stepping.
2. **Magnitude surfaces persistent features, not event features (the deeper cause).** Of the top-50
   features by activity, ~15 were collision-correlated, ~22 flat, ~13 anti-correlated — and the
   collision structure is highly reproducible (Δ collision-vs-air Spearman **+0.977** across
   episodes). But the **single most-active feature, #1364** — the top of the magnitude ranking the
   panel shows by default — is the most strongly *anti*-collision feature (mean activation 5.6 in
   mid-air vs 0.8 at collision): a ball-tracker that goes quiet at the paddle. A user watching the
   top-firing feature during a collision sees it stay flat. Real collision detectors (#1199, #120)
   exist, fire as discrete spikes locked to collisions (4–6× event lift, reproducible), and are
   robust — they just rank *below* the persistent tracker by magnitude, because brief events
   accumulate less total activity.

So the "decoupling" was a small display bug plus a **ranking mismatch**: magnitude shows the user a
feature whose semantics don't match the event they had in mind. Temporal stability surfaced the
collision detectors better (4 of its top 5 were collision-correlated) — evidence, not just
principle, that magnitude is a poor *semantic* importance signal here. (Test 5 also caught a
methodological subtlety: a feature can score high on a *group-mean* event contrast merely by being
active during a block that overlaps the events; per-event timing disambiguates, and demoted one
false positive.)

## 4. Clean negative result: causal importance doesn't survive as a ranking

Causal importance — rank features by the token-divergence their ±intervention induces in an
imagined rollout — was the most defensible axis in principle and the most novel. Building it
honestly unwound it in three stages:

1. **A magnitude confound.** The injection was magnitude-relative (`scale × activation ×
   direction`) — correct for the *live intervention tool* but wrong for a *ranking*: bigger
   perturbations cause bigger downstream change, so the score covaried with activation
   (`corr +0.37`; the top causal features were the top-activation features in order). This is
   invisible in the per-feature math; it only shows at the population-level cross-check.
2. **The fix exposed a deeper limitation.** Fixed-norm injection (`scale × unit_direction`, same
   magnitude for every feature) removed the confound (`corr → ~0`) — but the scores collapsed and
   re-runs came back *bit-identical*. The cause: the Atari env reset plus a confident policy are
   **effectively deterministic**, so every run — even re-seeded — scored against *one* game state.
   The confound had been masking a single-state measurement. Genuine diversity required
   time-sampling states along a playthrough.
3. **With diverse states, the signal is real but not robust enough to rank.** A true collision
   detector (#120) rose to the top, and #1364 is causal rank 1 *consistently with the confound gone*
   — confirming it drives the imagined dynamics without being a visible event (a genuine
   prediction/model-internal feature). But the decisive test, **cross-set reproducibility across two
   disjoint state-sets**, only climbed with diminishing returns: **1 → 5 → 18 states gives Spearman
   ~0 → +0.33 → +0.49**, never clearing a usable bar, and the pipeline is CPU-dispatch-bound so more
   states is impractical.

**Decision (pre-committed: integrate iff cross-set Spearman ≥ 0.6): +0.49 < 0.6 → not shipped as a
discovery ranking.** Causal importance remains a *characterization* tool — it reliably flags the
top handful of causally-potent features (#1364, #120) for per-feature analysis — but it cannot
rank ~2K features reproducibly. Two of three axes shipped; the third was demoted on evidence.

## 5. Transferable findings

Full detail in [FINDINGS.md](FINDINGS.md); the load-bearing ones:

- **Magnitude surfaces persistent features, not event features** — for event-detector discovery,
  magnitude is the wrong sort key; event-conditioned contrast or temporal stability is better.
- **Causal-importance pipelines must use magnitude-independent injection**, or they silently
  collapse to a magnitude ranking. Only visible at the population-level cross-check.
- **A deterministic env + confident policy means "more seeds" doesn't diversify the seed state** —
  a causal score from one deterministic rollout is a per-state probe, not a feature ranking;
  verify state diversity with two disjoint state-sets.
- **Autoregressive rollout pipelines are CPU-dispatch-bound, not GPU-bound** — the WM is rolled one
  token at a time in a Python loop, so a bigger GPU barely helps and parallel jobs contend for CPU;
  batching the rollouts is the real speedup.
- **A one-frame offset between a rendered frame and its activations** reads as feature–event
  decoupling; one-message bundling guarantees they travel together, not that they describe the same
  step — measure the lag.
- **Single measurements are provisional**: the project's recurring discipline (≥2 runs / 2 seeds /
  2 episodes before a conclusion) caught the magnitude confound, the single-state fragility, and a
  block-confounded "event detector."

## 6. What shipped vs. what didn't

| | Status |
|---|---|
| v2 canvas tool, pinned cards, search, multi-feature intervention | shipped |
| Temporal-stability discovery ranking | shipped |
| Magnitude (firing) ranking | shipped (as the contrast/baseline) |
| Autointerp labels | built, not run (needs API key; UI shows "unlabeled") |
| Causal-importance discovery ranking | **not shipped** — demoted to characterization tooling |
| Display/activation frame sync | bug found and fixed |

## 7. Process retrospective (honest)

The research conclusions are solid; the path to them was not efficient, and that's worth recording.

- **Compute choices cost time.** Much of the causal investigation ran on slow local MPS, then on a
  cloud GPU that was **contended and never the bottleneck** (the pipeline is dispatch-bound). Moving
  to a bigger GPU didn't help the per-call rate; running two jobs in parallel *hurt* (CPU
  contention). The real fix — batching the rollouts — was identified but not implemented.
- **I lost progress visibility** by setting the cache checkpoint interval to the end of an
  ~80-minute run, leaving no way to read partial progress. That compounded a string of **overly
  optimistic time estimates** that I should have stopped giving.
- **A pod was left idle and billing** because I went local without making that loud enough after
  being handed SSH; that was avoidable.
- What went right: the verification discipline. Refusing to integrate the causal toggle without a
  population-level cross-check is exactly what caught the confound; the pre-committed Spearman
  threshold turned "it feels noisy" into a clean go/no-go.

If repeated: batch the rollouts first, keep frequent checkpoints, and decide compute placement once
and loudly.

## 8. Open threads

- **Batch the causal rollouts** to make many-state runs cheap; then re-test whether causal
  importance crosses the robustness bar with enough states.
- **Run the autointerp pipeline** (cost <$5) so each ranking's top features can be read by label.
- **Multi-frame priming** for livelier imagined rollouts (single-frame priming gives near-static
  dynamics — a separate documented limitation).
- **Cross-game / cross-layer replication** — everything here is Breakout, layer 5.
