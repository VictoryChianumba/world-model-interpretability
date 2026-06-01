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

Trained TopK SAEs on layer 5 residual stream activations from ~5K Breakout frames. Got an interpretable SAE (L0 ≈ 15, sharp magnitude falloff), but discovered that the live-feature-list UX borrowed from LLM tools didn't work — features kept switching as the rollout advanced, no persistent identity, no way to track or compare features across frames.

## Part III — The intervention design that didn't measure what it claimed

The first intervention design rendered a "reconstruction" of the next frame and compared baseline vs. intervened. Turned out to be a tokenizer round-trip, not a prediction. Fixed to use the actual world-model imagination path.

Then discovered that single-frame diffs can't show dynamics — bricks moving is visible in one frame, paddle behavior is not. Redesigned around N-step imagined rollouts with token-divergence trajectories as the main signal.

Two structural findings dropped out: single-frame-primed imagination produces near-static rollouts (the model doesn't know velocity from one frame), and pixel-space state extraction from 16-token frames is fundamentally limited (the ball is too small to recover). Token divergence became the load-bearing metric.

## Part IV — A different kind of UI

v1's panel-grid layout was functional but ergonomically wrong. Studied Neuronpedia and EffectVis to understand the design language of LLM SAE tools; the central pattern is features have stable identity, supported by feature pages (Neuronpedia) or pinned cards on a canvas (EffectVis).

Redesigned around an EffectVis-style canvas with pinned feature cards, search-based discovery, multi-feature interventions, and a demoted "Model Internals" tab for the older panels. First implementation attempt failed to land the canvas; second attempt landed it. Currently functional, missing aesthetic polish and the rich-feature-card content layer.

## Part V — The open question

Top-K-by-activation-magnitude is the LLM SAE convention but doesn't transfer cleanly to world models, where frames are highly correlated and the underlying state changes slowly. The right notion of "feature importance" for world models is an open question and the current research direction.

Working hypothesis: temporal stability (features that fire consistently across many frames) combined with causal importance (features whose intervention measurably changes rollouts) is a better signal than raw magnitude. Implementing both is the next phase. Documenting them is the research output.

## Where this sits

A working interpretability tool for IRIS-class world models, structurally correct and functionally complete for its core experiment (intervene on a feature, see imagined rollouts diverge over 20 steps). The methodological findings — about tokenizer compression, argmax decoding limits, single-frame priming, and the inadequacy of LLM-style feature importance — are the research output, more than the tool itself. The tool is the substrate that made the findings visible.
