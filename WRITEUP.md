# WM Visualizer: an interpretability tool for IRIS-class world models

Personal project, exploratory scope. M1 + occasional rented A100. Built as a follow-up to a prior factored-SAE project, with two stated goals: (1) build the kind of interactive feature-exploration tool that exists for LLMs (Neuronpedia, EffectVis) but not for world models, and (2) characterize what's different about doing SAE-based interpretability on a world model substrate vs. an LLM substrate.

## Part I — Original Python visualization (`iris_interpretability.py`)

Starting point was a pygame + matplotlib live viewer of IRIS running on Breakout. Three panes: game frame, per-head attention heatmaps (2×2 grid), residual-stream norm bars across layers. Controls via keyboard (`[`/`]` to switch layers, Enter to reset). Hydra-composed agent loading. Real-time interactive loop, not static plotting.

**What this established:** the hook infrastructure for extracting world-model residual-stream activations, the single-block no-KV-cache forward pass for consistent 17×17 attention, the Hydra pattern for loading the agent. These all survived into later versions and are the load-bearing parts.

**What it didn't have:** any SAE, any intervention, any persistent state. It was a debug viewer.

## Part II — Migration to FastAPI + React web frontend (v1)

Reasons for the migration: web stack would give better visual options (multiple synchronized panels, custom interactions, real-time updates via WebSocket), and the matplotlib-via-pygame rendering was constraining.

**Architecture:** FastAPI backend running the IRIS model with the existing hook infrastructure, WebSocket fan-out for per-frame activations to the browser, REST control endpoints for pause/resume/restart/step/loop, React frontend with visx/D3 for the visualizations.

**What survived from the Python version:** `IrisHookExtractor` (hooks attached to WM residual stream), the single-block forward pass, the three conceptual visualizations (now `GameFrame`, `AttentionHeatmap` + `MorphingGraph`, `ActivationNorms`), the Hydra agent-loading pattern.

**What was rewritten:** the entire rendering layer. pygame and matplotlib were replaced by canvas + visx/D3 in the browser. Keyboard controls became REST `POST /control`.

**The v1 layout:** four panels — game frame top-left, attention (heatmap or graph toggle) middle, activation norms + SAE features + log right column, intervention rollout bottom-left.

## Part III — SAE training pipeline

Trained TopK SAEs on layer 5/6/7 residual stream activations from Breakout rollouts. ~80k vectors total (22 episodes / 4,717 frames × 17 token positions per frame). 100 epochs, L1=2.0.

**Results:** L0 ≈ 14-15 at all three layers (interpretable band is 10-50, vs. the placeholder's ~960), reconstruction MSE around 20. Top-feature magnitudes fall off sharply (3.5 → 2.7 → 2.1 → 1.2 → 0.5) instead of the placeholder's flat 2.2-2.7. SAE is real and behaving like a normal SAE.

MPS worked cleanly — 2× speedup over CPU on the activation harvest bottleneck (98s vs 195s). MPS device was previously flagged as untested in CLAUDE.md; that caveat is no longer accurate.

### Methodological note: 5K frames is small

Initial harvest produced ~80K vectors from ~5K frames. The literature rule for clean feature recovery is 100:1 to 1000:1 sample-to-feature ratio; with 2K SAE features, 80K vectors gives ~40:1. Borderline. More episodes would improve feature quality; generating them is a few hours of background time on M1 since IRIS can imagine its own rollouts. This is a cheap improvement that hasn't been done yet.

## Part IV — Intervention rollout endpoint

The first intervention design was a single-frame diff: pause, perturb feature N, decode the next-frame token reconstruction, compare to baseline. This turned out to be the wrong observable.

### Bug: "reconstruction" panel was measuring tokenizer round-trip, not prediction

The v1 panel labeled "imagined next frame" was actually `tokenizer.encode(current_obs)` → `tokenizer.decode(...)` — an autoencoder round-trip measuring tokenizer fidelity, not world-model dynamics. The genuine imagined-next-frame path (`world_model_env.py`'s `step(action)`: 1+K autoregressive WM passes with KV cache, sampling each next-obs token from `logits_observations`) exists in IRIS but wasn't being used.

**Fix:** wire `world_model_env.step` into a single-step imagine endpoint, restricted to the dominant chart's reconstruction (paused-only).

### Bug: single-frame diff is the wrong observable for testing dynamics

Even with the correct imagined-next-frame, comparing one baseline vs. one intervened frame can only show static differences (bricks moved). It cannot show dynamic differences (paddle behaves differently over time) because dynamics emerge over multiple frames. The single-frame design was unable to test the actual research question.

**Fix:** N-step imagined rollout endpoint. Pause at state S, run N=20 imagined frames under baseline, run N=20 imagined frames under intervention with identical action sequence (replayed from baseline), produce side-by-side scrubbable frames + trajectory line charts (paddle x, ball x/y) showing whether and how the imagined dynamics diverged.

**Cost:** ~17 WM passes per imagined frame × 20 steps × 2 conditions × 2 seeds ≈ 2,700 passes ≈ 80s on CPU, ~40s on MPS. Paused-only operation; no live cadence.

### Methodological finding: single-frame-primed imagination is near-static

The N-step baseline rollout barely evolves over time (~0.8/255 mean pixel change across 20 frames). The world model, primed from a single observation frame, has no temporal context to know what's happening dynamically — it has position but not velocity. The imagined rollout is approximately a still scene.

This is a property of single-frame priming, not a bug. It limits what "intervention affects dynamics" can demonstrate: the divergence-from-baseline signal is real and measurable (2.1→4.5 token divergence scaling with intervention magnitude), but the baseline is "near-static" not "typical dynamic play," so the experiment measures "feature affects the static reconstruction" more than "feature affects how the game evolves."

### Methodological finding: tokenizer compression limits pixel-space state extraction

Plan was to extract paddle x and ball x/y from each imagined frame via pixel inspection, then plot trajectories. Pixel state extraction can't track the ball on lossy 16-token reconstructions (ball_x/y = None across all frames; paddle x stuck at 0.5; max brightness 0.56). The information isn't there to extract.

**Workaround:** token-divergence trace as the load-bearing measurement (how many of 16 obs tokens differ baseline-vs-intervened per step), with per-step mean pixel diff alongside. These always have signal and directly show "dynamics diverged." Paddle/brick pixel extraction kept but honestly flagged when it fails.

This is a transferable finding: pixel-space state extraction from world model rollouts is fundamentally limited by the tokenizer's spatial resolution. Anyone trying behavioral interpretability on IRIS-class models will hit this. Token-divergence is a more honest measurement.

### Methodological finding: argmax decoding limits intervention smoothness

The WM decodes each next-obs token by argmax over 512 confident logits — a hard step function. Even with a clean SAE, interventions need large magnitudes (~50 threshold, ~100 for clear effect) and the effect stays small and non-monotonic. Better SAE quality doesn't soften the decoder.

This means "smoother interventions" requires method changes (inject on all token positions, scale per-feature-magnitude), not architectural improvements to the SAE side. The decoder is the bottleneck.

## Part V — v1 frontend problems and v2 redesign

v1 was functionally complete (paused state → pick feature → set scale → run rollout → see baseline vs. intervened + trajectory charts) but had structural UX problems.

### Problem 1: top-K-by-activation list churning under the user

The SAE feature panel showed "top 10 firing features for the current frame," refreshing as the rollout advanced. By the time a user read the list and decided which feature to intervene on, the list had changed. The list was unstable as a picker.

This is a deeper problem than UI: top-K-by-activation magnitude is the LLM SAE convention (each token is a discrete decision, high activation = strong contribution), but it transfers poorly to world models where frames are highly correlated and the underlying state changes slowly. Ranking by magnitude amplifies noise in the tails of the distribution rather than reflecting feature importance.

This is the central open methodological question for the project. Possible alternatives: temporal stability (rank by firing consistency across frames), causal importance (rank by intervention effect on rollouts), sensitivity (rank by scaling response), game-state correlation (rank by predictability of ground-truth state). Currently not implemented; flagged as the next research direction.

### Problem 2: features had no persistent identity

A feature in the v1 list existed for the current frame and then was gone. There was no way to bookmark a feature, track it across many frames, or build up a working set of features to compare.

This is the load-bearing UX issue. Neuronpedia solves it with feature pages (each feature has a persistent URL with activation examples, labels, dashboards) and Lists (user-curated bookmarks). EffectVis solves it with feature cards on a canvas. Both treat features as things with stable identity; v1 treated them as transient activations.

### Problem 3: vestigial panels

Attention heatmap/graph and residual norms panels were valuable during initial exploration but didn't earn their screen real estate in the intervention workflow. They're general-purpose interpretability views; once the user knows they want intervention effects on dynamics, they're decoration.

### v2 redesign decisions

- Canvas with pinned feature cards as the main surface (EffectVis pattern).
- Search/jump-to-feature-ID as the primary discovery interface.
- Top-firing list demoted to a side discovery panel; click "+pin" to move to canvas.
- Multi-feature intervention: each pinned card has its own scale slider; rollout sums contributions across all non-zero cards.
- Attention/norms demoted to a "Model Internals" tab, hidden by default.
- Rollout panel underneath the canvas with steps/seeds configurable inline (defaults: 20 steps, 2 seeds).

### v2 build outcome

First implementation attempt produced a polished v1 layout with new toggles, not the v2 canvas redesign. Pushed back on Claude Code; second attempt produced the actual canvas redesign with search, pinning, and the rollout panel below.

**Currently working:** search by ID or label, pin/unpin features to canvas, multi-card intervention, N-step rollout with token divergence trajectory. **Currently missing or unverified:** rich feature cards (no top-activating-frame thumbnails, no sparklines yet), autointerp labels (most features show "unlabeled"), Excalidraw-style canvas annotations, session save/restore, polished visual design (typography, spacing, color system).

The structural redesign is correct. The aesthetic and content layer is unstarted.

## Part VI — Feature importance investigation (in progress)

Open question identified during v2 use: top-K-by-magnitude doesn't transfer cleanly from LLMs to world models. Need a different importance ranking adapted to world model substrates.

Candidate alternatives (none implemented yet):

- **Temporal stability:** rank by firing rate variance over the last N frames (inverted — lower variance = higher rank). Features active across many consecutive frames are more likely to be about a stable concept than features that flicker.
- **Causal importance:** ablate or scale each feature, measure rollout divergence, rank by effect magnitude. Most defensible definition; expensive (one rollout per feature, ~80s on CPU each).
- **Sensitivity:** how does intervention magnitude correlate with rollout divergence per feature? Cheaper version of causal importance.
- **Game-state correlation:** regress each feature's activation against ground-truth state variables. Limited by tokenizer compression for ball position; might work for paddle and brick state.

Working hypothesis: temporal stability × causal importance, combined, will produce a more useful ranking than activation magnitude alone. Causal importance is the most novel and most research-publishable of these — "causal importance ranking of SAE features in a world model" doesn't have published prior art that I've found.

Plan: implement temporal stability as the immediate fix (small backend change, makes the discovery panel less churny). Implement causal importance as an offline pipeline (one rollout per feature, ~44 hours on CPU or ~22 on MPS, runs overnight). Surface both rankings as alternate views in the discovery panel. Document both as findings.

### Implementation — temporal stability (done)

Built as a small engine change: a rolling 60-frame window of the full per-feature activation vector (`_sae_history`, written each frame in `_compute_sae_features`), exposed via `GET /ranking/stability`. The discovery panel gained a metric toggle (`firing` / `stable` / `causal`), and "stable" polls the endpoint live (1.5s) so it reflects the recent window.

**Design decision — rank by coefficient of variation, not raw variance.** The literal spec ("lower variance = higher rank") is degenerate: a permanently-OFF feature has variance 0 and would top the ranking while meaning nothing. Two changes fix it: (1) gate on a firing-rate floor (default 0.2) so features must actually fire in the window to be ranked, and (2) rank by ascending coefficient of variation (std/mean) rather than raw variance, which is scale-invariant — a feature that fires steadily at magnitude 0.5 and one that fires steadily at 3.0 are both "stable," and CV says so while raw variance would over-rank the small-magnitude one. The display score is `1/(1+CV)` so higher reads as more stable, matching the firing-magnitude bars' direction.

This is the first concrete instance of the project's thesis: importing the LLM convention naively (rank by a magnitude statistic) produces a broken ranking on the world-model substrate, and the fix requires a substrate-aware metric. The dead-feature degeneracy is logged in FINDINGS.

### Implementation — causal importance (done; partial sample measured)

Built as an offline pipeline: `scripts/causal_importance.py` → `causal_L{layer}.json` → `GET /ranking/causal`, surfaced as the discovery panel's third toggle. For each feature it runs ±scale intervention rollouts against a shared baseline (same seed, same frozen action sequence) and scores by **mean token divergence** — token-only, no pixel decode, so it is far cheaper than the live `/rollout`. Injection is magnitude-relative (scale × the feature's own seed-state activation), matching the live intervention semantics. Built to sample and resume; `pos`/`neg` per-sign means are kept so an asymmetric feature is visible rather than averaged away.

**Sample run — two independent runs, 24 most-active features, 2 seed states, 10 steps, ±5, MPS, ~9 min each:**

- **Causal importance vs activation magnitude:** Spearman **+0.56** and **+0.64** across the two runs. Magnitude is a weak-to-moderate proxy for causal effect — within the *same* high-firing set, some features move the rollout hard (token divergence up to ~15 of 16) and others barely move it (~2 of 16). The single most-active feature (#1364) was also the most causal in both runs, but mid-ranking diverged. So magnitude predicts the very top but not the order.
- **Run-to-run robustness:** the two runs shared only **18 of 30** distinct features — "top-K by activation" itself selected different features run-to-run (different seed states), the churn problem in another guise. On the shared set causal scores correlated at Spearman **+0.56**, with top-5 causal overlap **4/5**. The very top is reproducible; the tail is not from a single run.
- **Sign asymmetry:** per-feature |effect(+5) − effect(−5)| averaged ~0.8 tokens — features respond asymmetrically to amplification vs suppression.

**Caveat (scope, stated honestly):** this sample scores only the high-firing head (top-24 by activation), so the magnitude-vs-causal correlation is measured *within* active features and is biased **upward** — across all ~2K features, including the low-firing majority where the two diverge most, agreement would be lower. The sample is enough to show the pipeline works and that magnitude is an imperfect proxy even among active features; it is not the full-feature result. The full run (all features, more seeds) is the next step. Per single-measurement discipline, every number here is from a 2-run / 2-seed sample and should be read as provisional beyond the top handful.

**Which ranking is "more interpretable"?** Not yet answerable on this evidence — it needs the autointerp labels (also built, not yet run) so a human can read the top features of each ranking side by side. What the sample *does* show: the three rankings genuinely disagree (firing is unstable frame-to-frame and run-to-run; causal only moderately tracks magnitude), so they are not redundant views — picking the ranking is a real choice, which is why all three are exposed.

### Feature characterization test battery

A diagnostic batch run to resolve an observation made while using the tool: SAE features appearing *decoupled* from on-screen events (firing while the ball is mid-air, flat during paddle-ball collisions). Five tests distinguish the candidate causes — (a) display-pipeline timing, (b) extraction noise, (c) wrong feature labels, (d) features that are semantically real but track something other than the expected event, (e) genuinely broken/polysemantic features. The tests are diagnostic; they do not change the running tool. Single-measurement discipline applies: results that depend on sampling specific frames or rolling out across an episode require ≥2 independent runs.

#### Test 1 — frame/activation synchronization

**Question.** Are the displayed game frame and the SAE activations shown beside it from the same model forward pass?

**Method.** (1) Added a monotonic `frame_index` to every emitted message plus `console.debug` render logging in the frontend. (2) `scripts/diagnostics/test1_frame_sync.py` replicates the engine's per-step sequence and measures the lag between the displayed frame and the obs the activations are computed from: downsample both to 64×64 grayscale and, over candidate lags L ∈ [−2, +2], find the L minimizing mean |display[t] − obs[t+L]|. Run on 2 independent episodes. (3) A 600-frame WebSocket capture checked message-level co-delivery and `frame_index` monotonicity.

**Result — two findings, opposite in sign:**

- *No multi-message drift.* The game frame and activations are not separate messages — they are bundled in one `FrameData` and rendered from one React state update. Over 600 streamed frames every message carried both, with `frame_index` strictly increasing 1→600. The display-pipeline-timing hypothesis (a) is ruled out as a *multi-message / render-batch* problem.
- *A real one-frame content offset, now fixed.* Within that single message the frame was captured **after** `env.step()` while the activations describe the **pre-step** obs — so the displayed frame led its own activations by exactly one step. The lag diagnostic measured best-lag **+1** for the old post-step capture and **0** for the pre-step capture, identically in both episodes (minimum mean-abs-diff 0.00376 at the aligned lag vs 0.00428 one step off — the whole lag table shifted by one). Fixed by capturing the display frame before `env.step()`; `frame_index` now labels the shared step.

**Interpretation.** This offset is exactly the kind of artifact that reads as "features don't fire during the on-screen collision": when the displayed frame shows the collision, the activations beside it are from the frame just *before* it. It is a genuine bug (not perception bias) and a *partial* explanation of the observation — but only partial: a one-frame shift mis-attributes by one frame, so it cannot by itself account for features that stay flat across an entire multi-frame collision window. Whether features align with events *at all*, after this fix, is what Tests 3 and 5 measure.

**Done:** frame–activation drift detected (one-frame semantic offset) and fixed by pre-step capture; multi-message/render drift ruled out across 600 frames; 2 episodes agree.

#### Test 2 — activation determinism

**Question.** Given the same input frame, does the SAE produce the same activation vector on repeated forward passes, or is there extraction noise?

**Method.** `scripts/diagnostics/test2_determinism.py` replicates `_compute_sae_features` exactly (encode obs → WM forward with a residual hook on the SAE layer → normalise the action-token vector → SAE.encode) on a *fixed* `(obs_tokens, action)` input — the action fixed too, since the SAE reads the action-token position — repeated 10× on each of 3 distinct game states. Diff every repeat against the first across all 2048 features.

**Result.** Bit-exact deterministic: max |Δ| = 0.0, mean |Δ| = 0.0, and 0 of 2048 features with |Δ| > 1e-6, on every frame and repeat. `world_model.training`, `tokenizer.training`, and `sae.training` all `False` (eval mode confirmed, dropout off). Active feature counts at the action token were 8–18 of 2048, consistent with the SAE's measured sparsity (L0 ≈ 14–15).

**Interpretation.** Extraction noise (case b) is ruled out — repeated extraction on identical input is exactly reproducible, so any feature–event decoupling is a property of *what the features represent*, not jitter in measuring them. Caveat: verified on CPU; MPS/CUDA can introduce nondeterministic reductions, but the live extraction runs the same code path and the eval-mode flags hold regardless of device. With Tests 1 and 2 done, the two *mechanical* explanations (display-pipeline timing beyond the fixed one-frame offset, and extraction noise) are eliminated; the remaining candidates (c/d/e — wrong labels, features tracking something other than the expected event, or broken/polysemantic features) are about feature *semantics*, which Tests 3–5 measure directly.

**Done:** diff statistics (all zero) recorded; eval-mode confirmed; extraction noise ruled out.

## Part VII — Open threads

- Feature importance pipeline (temporal stability + causal importance), as above.
- Autointerp pipeline (vision-LLM on top-activating frames per feature, produces concept labels). Cost: <$5 for 2K features. Not yet implemented.
- Sparklines on feature cards (firing history over recent frames).
- More episodes: 80K vectors is borderline; doubling/tripling is a few hours of background time.
- Excalidraw-style canvas annotation layer (arrows between cards, text notes).
- Session save/restore (multiple named experiment pages).
- Method change: inject intervention on all token positions, scale per-feature-magnitude, to soften the argmax-decoder bottleneck.
- Layer sensitivity: SAEs trained at layers 5/6/7; project mostly uses layer 5. Whether features at different layers correspond to different abstraction levels is open.
- Cross-game replication: SAE only trained on Breakout. Alien agent loads but has no SAE.
