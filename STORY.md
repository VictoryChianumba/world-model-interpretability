# The narrative arc — WM Visualizer

A short-form companion to [WRITEUP.md](WRITEUP.md).

## The spine

One question, escalating: can SAE features in a world model's residual stream be discovered, interpreted, and used to causally control the model's imagined rollouts, on a substrate small enough to iterate on locally?

## Part 0 — Why a world model

After a previous project on factored SAEs for LLM concept manifolds closed with honest-but-narrow results, the next instinct was to ask whether the manifold-discovery techniques transfer to world models. IRIS — a transformer-based world model trained on Atari — became the substrate: small enough to run on M1, structured enough to have interesting representations, paired with a tiny known game state (Breakout) that provides ground-truth for what features should exist.

## Part I — From debug viewer to interactive tool

The starting point was a pygame visualizer of IRIS running Breakout with attention heatmaps and residual norms displayed live. Useful for debugging, not useful for interpretability — no SAE, no intervention, no way to do anything with the activations.

Migrated to a FastAPI + React web stack so the rendering layer could support the interactivity needed for SAE-based interpretability work. The model logic, hooks, and forward-pass infrastructure all survived; the rendering layer was rewritten from scratch.

## Part II — Training an SAE and discovering what didn't work

Trained a ReLU/L1 SAE on layer 5 residual stream activations from ~5K Breakout frames. Got an interpretable SAE (L0 ≈ 15, sharp magnitude falloff), but discovered that the live-feature-list UX borrowed from LLM tools didn't work — features kept switching as the rollout advanced, no persistent identity, no way to track or compare features across frames.

## Part III — The intervention design that didn't measure what it claimed

The first intervention design rendered a "reconstruction" of the next frame and compared baseline vs. intervened. Turned out to be a tokenizer round-trip, not a prediction. Fixed to use the actual world-model imagination path.

Then discovered that single-frame diffs can't show dynamics — bricks moving is visible in one frame, paddle behavior is not. Redesigned around N-step imagined rollouts with token-divergence trajectories as the main signal.

Two structural findings dropped out: single-frame-primed imagination produces near-static rollouts (the model doesn't know velocity from one frame), and pixel-space state extraction from 16-token frames is fundamentally limited (the ball is too small to recover). Token divergence became the load-bearing metric.

## Part IV — A different kind of UI

v1's panel-grid layout was functional but ergonomically wrong. Studied Neuronpedia and EffectVis to understand the design language of LLM SAE tools; the central pattern is features have stable identity, supported by feature pages (Neuronpedia) or pinned cards on a canvas (EffectVis).

Redesigned around an EffectVis-style canvas with pinned feature cards, search-based discovery, multi-feature interventions, and a demoted "Model Internals" tab for the older panels. First implementation attempt failed to land the canvas; second attempt landed it. Currently functional, missing aesthetic polish and the rich-feature-card content layer.

## Part V — The open question

Top-K-by-activation-magnitude is the LLM SAE convention but doesn't transfer cleanly to world models, where frames are highly correlated and the underlying state changes slowly. The right notion of "feature importance" for world models is an open question and the current research direction.

Working hypothesis: temporal stability (features that fire consistently across many frames) combined with causal importance (features whose intervention measurably changes rollouts) is a better signal than raw magnitude. Both were implemented and tested — and only one survived:

- **Temporal stability** (kept) — ranking by raw variance is degenerate (dead features have zero variance and look "perfectly stable"), so the metric needs a firing-rate gate and a scale-invariant coefficient of variation. Ships as the second discovery axis.
- **Causal importance** (demoted) — looked promising until integration triggered a verification pass that unwound it. First, a magnitude confound: the injection scaled by each feature's own activation, so "most causal" was "most active" in disguise. Fixing that (magnitude-independent injection) exposed a deeper problem — the env and policy are effectively deterministic, so the score reflected *one* game state; re-running, even re-seeding, gave bit-identical results. Forcing genuine state diversity helped but plateaued: cross-set reproducibility climbed 1→5→18 states as ~0 → +0.33 → +0.49 Spearman, never clearing the 0.6 bar. So causal importance was **not shipped as a discovery ranking** — it stays a characterization tool that reliably flags the top handful of causally-potent features (the air-tracker #1364, the collision detector #120) but can't rank all 2K reproducibly.

The honest outcome: of three proposed importance axes, **two shipped** (firing, stability) and one was demoted after its confound and single-state fragility were measured. Finding *why* causal didn't make it — magnitude coupling, deterministic seed states, a dispatch-bound pipeline a bigger GPU can't rescue — is the more transferable result than a third ranking would have been. Negative results, documented, are the research output.

## Part VI — Diagnosing the decoupling

The open question turned concrete through a complaint while using the tool: SAE features looked *decoupled* from on-screen events — firing in mid-air, flat during paddle-ball collisions. A five-test diagnostic battery was run to find out whether the features were broken, mislabeled, or just being mis-ranked. The first three tests resolved it:

- A real but small **display bug**: the game frame was captured one step after the activations it was shown beside, so the frame led its own activations by one frame (fixed).
- Extraction is **bit-exact deterministic** — no measurement noise.
- The decisive finding: features are **not** decoupled. Clean collision detectors exist and are highly reproducible (Δ collision-vs-air ranking correlates +0.977 across episodes). But the *most-active* feature — the one the magnitude ranking puts at the top — is an air-flight ball-tracker that goes quiet at the paddle. The user was watching the top of the firing list during a collision and seeing a feature whose job is to track the ball in flight, not detect the hit. The "decoupling" was a ranking mismatch, not absent structure.

So the central question is now answered, not just framed: activation magnitude *is* a poor importance signal here — not because features lack meaning, but because magnitude ranks persistent features over event features, and the event features are exactly what a human scanning for "the collision feature" expects to see. Temporal stability surfaced the collision detectors better (4 of its top 5 were collision-correlated). The substrate-adapted rankings earn their place on evidence, not just principle.

## Where this sits

A working interpretability tool for IRIS-class world models, structurally correct and functionally complete for its core experiment (intervene on a feature, see imagined rollouts diverge over 20 steps). The methodological findings — tokenizer compression (even the 64×64 model input loses the ball), argmax decoding limits, single-frame priming, a one-frame display/activation offset, and above all that magnitude surfaces persistent features rather than event features — are the research output, more than the tool itself. The tool is the substrate that made the findings visible; the diagnostic battery is what turned "the features feel decoupled" into "magnitude ranks the wrong features, and here is the one the user was watching."
