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
