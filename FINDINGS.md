# Findings

Transferable methodological observations from building the WM Visualizer. These apply beyond this specific project to anyone doing SAE-based interpretability on IRIS-class autoregressive world models.

## Tokenizer compression limits pixel-space state extraction

IRIS tokenizes 84×84 game frames into 16 discrete tokens via a VQ-VAE. Decoded reconstructions retain coarse spatial structure (paddle position, brick layout) but lose small or fast-moving objects (the Breakout ball). Pixel-space state extraction — finding where the paddle and ball are in each imagined frame — works for large persistent objects and fails for small ones. Max brightness on the ball region drops to ~0.56 after the round-trip; subpixel-scale information doesn't survive.

**Implication:** behavioral interpretability on IRIS-class models can't rely on extracting object positions from imagined frames at arbitrary precision. Token-level divergence (how many of 16 obs tokens differ between baseline and intervened rollouts) is more honest and always-available. Pixel diff is a useful secondary signal. Per-object pixel extraction is fine for large objects, fails silently for small ones — annotate failures explicitly.

## Single-frame-primed imagination is near-static

IRIS's `world_model_env.step` autoregressively generates next frames given a current observation and an action. Primed from a single observation frame, the model has no temporal context — it knows position but not velocity. Imagined rollouts evolve very slowly (~0.8/255 mean pixel change across 20 frames in Breakout).

**Implication:** intervention experiments designed to test "feature affects dynamics" measure divergence from a near-static baseline. The intervention signal is real (token divergence scales with magnitude), but the dynamics being divergence-tested are subtler than typical gameplay. Multi-frame priming (feeding the WM several recent observations to establish velocity) would produce livelier rollouts but is scope creep into WM mechanics. Document the limitation honestly.

## Argmax decoding limits intervention smoothness

IRIS decodes each next-obs token by argmax over 512 confident logits — a hard step function. Even with a high-quality SAE (L0 ≈ 15, sharp magnitude distribution), interventions need large magnitudes (~50 threshold, ~100 for visible effect) and the effect remains non-monotonic.

**Implication:** improving SAE quality doesn't soften the intervention response. The bottleneck is the decoder. Smoother interventions require method changes — injecting on all token positions instead of one, scaling per-feature-magnitude, or modifying the decoding step. These are method-development projects in their own right, not tuning passes.

## Activation magnitude is a poor importance signal for world models

LLM SAE tools rank features by activation magnitude on the current token, which works because each token is a discrete decision point and high activation correlates with strong contribution to that decision. World models break this assumption: frames are highly correlated, the underlying state changes slowly, and "top firing right now" is dominated by noise in the tails of a slow-moving high-dimensional state.

Better candidates: temporal stability (features that fire consistently across many frames), causal importance (features whose intervention measurably changes rollouts), or sensitivity (features whose scaling response is monotone). None are implemented yet; activation magnitude remains the discovery default and produces a churning, hard-to-navigate UI.

**Implication:** importing LLM SAE conventions directly into world model interpretability tooling produces broken UX. Importance metrics need to be adapted to the substrate.

### Edge case — "rank by low firing-rate variance" is degenerate without a firing-rate floor

When implementing temporal-stability ranking (rank features that fire consistently above features that flicker), the obvious metric — lowest variance of activation over a window — is degenerate. A permanently-OFF feature has variance exactly 0, so a naive low-variance ranking surfaces dead features at the top: maximally "stable," entirely meaningless. SAEs have many dead/rare features, so this isn't a corner case; it dominates the ranking.

Two fixes, both needed: (1) gate candidates on a firing-rate floor (a feature must be active in at least X% of the window, default 20%, to be ranked at all), and (2) rank by coefficient of variation (std/mean) rather than raw variance. CV is scale-invariant, so a feature firing steadily at low magnitude and one firing steadily at high magnitude are scored equally stable — whereas raw variance conflates "low magnitude" with "stable" and over-ranks weak features.

**Implication:** any "consistency" or "stability" importance metric on a sparse-feature substrate needs an activity gate, because the silent majority of features are inactive and inactivity reads as perfect stability. State the gate explicitly; it changes which features the ranking is even considering.

### A step-and-render loop can ship a frame one step out of sync with its own activations

In an interactive world-model viewer the natural loop is: act on obs → step the env → read activations → render. If the displayed frame is captured from the environment *after* the step (e.g. via the env's current rendered observation) while the activations were computed from the *pre-step* obs, the frame shown to the user leads its activations by one step. Measured here as a clean **+1 lag** — the argmin, over candidate shifts, of mean-abs-difference between the displayed frame and the activation obs (both downsampled to 64×64 gray) — reproducible across two episodes. Bundling frame and activations into one message guarantees they *travel and render together* but does NOT guarantee they describe the *same step*; the offset lived inside the single message.

This is invisible to static inspection and easy to introduce, and it manifests as apparent feature–event decoupling: features look like they fire "just before" the on-screen event. The fix is to capture the display frame from the same timestep the activations describe (before stepping the env).

**Implication:** when activations and a rendered frame are presented together as "the same moment," measure the frame↔activation lag explicitly — don't assume zero, and don't let one-message bundling stand in for same-step alignment. A one-frame offset is small but it is precisely the artifact a human reads as "the feature is decoupled from the event."

### Activation magnitude surfaces *persistent* features, not *event* features — which reads as "features are decoupled from events"

In a world-model SAE, ranking features by activation magnitude (the LLM-SAE default) is dominated by features that fire *continuously* — e.g. a feature tracking the ball through its entire mid-air flight — over features that fire *briefly at salient events* — e.g. a feature that fires only during a paddle-ball collision. The event feature accumulates less total activity (a few frames of contact vs. dozens of frames of flight), so it ranks lower. Measured on Breakout layer 5 (Test 3): of the top-50 features by activity, ~15 were collision-correlated, ~22 flat, ~13 anti-correlated; and the single most-active feature was the *most strongly anti-collision* feature in the set (mean activation 5.57 in mid-air vs 0.82 at collision) — a ball-tracker, not a collision detector. The collision detectors existed and were robust (Δ = μ_collision − μ_air reproducible across episodes at Spearman +0.977) but sat below the persistent ball-tracker by magnitude.

This is the mechanism behind a common complaint that "the SAE features don't correspond to the on-screen events." They can and do — but if the UI ranks by magnitude and the user expects the top feature to fire at a specific brief event, the top feature is instead whatever fires most *persistently*, whose semantics (continuous tracking) won't match a brief event. The feature looking "flat during the collision" is correct behavior for a mid-flight tracker.

**Implication:** for discovering *event-detector* features, magnitude is the wrong sort key — it ranks by duration×strength of firing, not by tie-to-an-event. Event-conditioned contrast (mean activation at event vs not, i.e. a Δ score) or temporal-stability ranking surfaces event/semantic features better. And before concluding "features are decoupled from events," check whether the *ranking* is showing you event features at all.

### Group-mean event correlation can be confounded by temporal blocks — verify with per-event timing

A feature can score high on "mean activation at the event vs not" (a Δ contrast) without being an event detector at all: if it is active during a sustained multi-frame *block* (a game phase or state) that happens to overlap the sampled event frames, the group mean is inflated. Caught directly here — feature #1773 scored as collision-correlated by group mean (Δ +1.13, Test 3) but had no reliable per-event activation lift (1.0× and 0.3× across two episodes, Test 5); its trace showed a sustained block of activation, and the collision frames sampled for Test 3 fell inside it. Genuine event detectors (#1199, #120) instead fired as discrete spikes locked to the event, with 4–6× per-event lift, reproducible across episodes.

**Implication:** "fires at the event" and "is active during a window containing the event" are different claims, and a group-mean contrast cannot tell them apart. Confirm event-feature claims with per-event timing (does activation spike *at* the event) and across independent episodes; the cross-episode check is what demotes the block-confounded features. Two methods that mostly agree but disagree on one feature is the expected, healthy outcome — the disagreement is the signal.

### Even the model's input resolution can be too coarse to label the object of interest

The Breakout ball is recoverable from the 210×160 human frame but **not** from the 64×64 obs the model actually consumes — at 64×64 the ball is sub-pixel and a brightness-blob extractor finds nothing across hundreds of frames. This extends the tokenizer-compression finding upward: it is not only the 16-token *reconstruction* that loses small objects, it is the model's 64×64 *input* itself. Any pixel-space event labeling for behavioral interpretability must therefore label from the original full-resolution frame, not the model obs (they depict the same moment, so the labels still align with the activations). Assuming the model input is fine-grained enough to label from is a trap.

### Causal importance only moderately tracks activation magnitude — and a single causal run is noisy

Two independent partial runs of causal-importance scoring (mean token divergence under ±5 intervention rollouts; 24 most-active features; 2 seed states each) on Breakout layer 5:

- **Causal vs magnitude:** Spearman +0.56 and +0.64. Activation magnitude is a weak-to-moderate predictor of causal effect even *restricted to high-firing features* — the firing head contains both high-impact and near-inert features (token divergence ranged ~2 to ~15 of 16). The single most-active feature was also the most causal in both runs, so magnitude predicts the very top but not the ordering below it.
- **Run-to-run robustness:** the two runs shared only 18 of 30 distinct features (top-K-by-activation selected different features run-to-run — the same churn that breaks the live "firing" ranking, now visible across runs). On the shared set, causal scores correlated at Spearman +0.56 with top-5 overlap 4/5. The top handful is reproducible; the rest is not from a single run.

**Implication:** causal importance is the most defensible ranking but the most expensive to estimate reliably. A single seed state or single run gives a trustworthy *top few* and an untrustworthy tail. Use ≥2 seed states (the pipeline default) and compare ≥2 runs before believing any mid-ranking ordering. And measuring magnitude-vs-causal agreement *only* on the top-K-by-activation set biases the correlation upward — the low-firing majority, where the two diverge most, is excluded; the honest comparison needs the full-feature run. This is the single-measurement discipline applied to a ranking rather than a scalar: the conclusion "feature X is more important than feature Y" needs more than one measurement, and how much more depends on how close X and Y are.

## Soft routing in MoE-style SAEs leaks signal into the gating mechanism

*(Carried from the previous factored-SAE project; relevant context here.)*

If an SAE variant uses soft (softmax) routing for differentiability, the dominant chart's reconstruction quality is not the right measurement target — the router blends information across multiple charts, producing reconstruction quality that's not attributable to any single chart's representation. The fix is hard top-1 routing during training, with evaluation restricted to the dominant chart's output.

**Implication for this project:** when designing MoE-style architectures for SAE-on-world-model variants (not currently done here, but a plausible next direction), hard routing should be the default and soft routing should be flagged as inviting confounds.

## Single-measurement results are provisional in both directions

Three times in the broader project — across the previous factored-SAE work and this one — a single-seed or single-measurement result misled the writeup. Once it was a protocol bug, once a noisy negative reported as a clean negative, once a noisy positive amplified by sampling luck.

**Implication:** at the scales and noise levels of solo-research interpretability work, anything reported from a single seed or a single experiment is provisional regardless of whether the result is positive or negative. Two runs minimum before drawing conclusions; more if the numbers are close to noise. The temptation to draw conclusions from a single run is strongest right after an honest reframe pass, when "negative" feels like the calibrated answer.

## Frontend redesigns benefit from referencing existing tools before iterating

Studying Neuronpedia and EffectVis after building v1 produced more design clarity in an afternoon than weeks of iterating on the panel layout would have. Existing tools in adjacent domains (LLM SAE interpretability) had already solved problems the world-model tool was reinventing: stable feature identity, search-first discovery, persistent pinning, dashboard-per-feature.

**Implication:** domain-novel doesn't mean UX-novel. The interpretability genre has design patterns; importing them is faster than reinventing them.
