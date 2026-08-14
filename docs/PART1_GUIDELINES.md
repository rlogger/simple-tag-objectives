# Part 1 guidelines (post-scaffold review)

Synthesized from three independent read-only reviews of the July 31 latent-strategy note against this repo:

- Claude Opus — roadmap / PDF compliance
- Grok — uncertainty & sequential belief
- GPT — architecture, leakage, reproducibility

These are **engineering/research guidelines**, not claims. Part 2 planners remain out of scope.

## Verdict in one line

The scaffold matches the PDF *recipe* (GRU-JEPA, L2, LayerNorm, no VICReg) but does **not** yet implement uncertainty-aware sequential opponent modeling: belief is an eval probe bolted onto a point latent, BC uses frozen `z`, and several leakage/confound paths would inflate headline numbers.

## Consensus P0 gaps

| Gap | Where | Why it matters |
|-----|--------|----------------|
| Logistic probe ≠ training-time belief | `mopa.belief.fit_latent_belief` | Right for eval; wrong for conditioning BC (label leak + closed K-simplex) |
| BC conditions on frozen point `z`, not `b_t` | `mopa.bc.bc_comparison` | Not sequential; not uncertainty-aware |
| `train_jepa_gru` does not return encode params | `mopa.encoders.train_jepa_gru` | Cannot run online `encode(params, prefix) → z_t` |
| Lengths / pad mask not wired to `capture_t` | `mopa.data.rollout_one_checkpoint`, `GRUEnc` | Post-capture stillness / padding become “behavior” |
| GRU-JEPA target = full-episode embedding | `encoders.train_jepa_gru` loss | Early `z_t` trained against future (capture/lava); overconfident anytime belief |
| `ckpt_seed` recorded, unused in splits | `ObjectiveDataset.ckpt_seed`, `episode_validation_mask` | Probe answers “this checkpoint” not “this strategy type” |
| Encoder often fit on full pool + global `z` centering | `train_jepa*`, `metrics.metrics` | Transductive / optimistic numbers |
| LayerNorm ≠ anti-collapse | `GRUEnc` LN then `Dense(lat)` | Cross-episode collapse possible; no variance/rank diagnostic |
| Co-trained prey per objective | MAPPO specialists + dataset | Encoder may fingerprint prey/checkpoint, not predator strategy |
| No end-to-end Part 1 driver / manifests | `scripts/` only has MAPPO | Not third-party reproducible |

## Unified ranked roadmap (≤10)

Do **1–4 before publishing any number**.

1. **Collapse diagnostics + bounded anti-collapse**  
   Log across-episode std, per-dim std, effective rank. Prefer unit-sphere `z` + cosine/L2-on-normalized prediction (keeps no-VICReg stance). Test: ARI > 0.5 on separable synthetic at default steps; rank floor.

2. **Checkpoint-grouped splits + train-only encode**  
   `checkpoint_validation_mask` / LeaveOneGroupOut on `ckpt_seed`. Fit encoder, probe, GMM, temperature **only** on train; encode val/test frozen. Report within-ckpt vs held-out-ckpt probe.

3. **Mask post-capture everywhere + shortcut controls**  
   Cap lengths at `capture_t`; drop BC steps `t ≥ capture_t`; pad-mask GRU. Required controls: probe on `survival_time` alone; length-matched prefixes.

4. **`scripts/run_part1.py` + online encoder API**  
   Specialists → dataset → GRU-JEPA (+ baselines) → belief → BC → JSON (metrics, seeds, git SHA, dependency pins). Return `{params, target_ema}` from `train_jepa_gru`. Put GRU in `evaluate_encoders`. Prefer future-window / later-prefix targets over always `t_full`.

5. **Belief-conditioned BC (mixture), probe stays eval-only**  
   Train specialists or type-heads without teacher-forced eval labels in the encoder. Online `b_t`; predict `π(a|s,b) = Σ_k b_t(k) π_k(a|s)`. Compare vs `π(a|s)`, `π(a|s,z_t)`, oracle `π(a|s,y)`. No feeding `fit_latent_belief` posteriors into BC training.

6. **Quarantine identity-JEPA**  
   Rename / flag `uses_eval_labels=True`; never on the SSL axis.

7. **Calibration + open-set**  
   Temperature on val; NLL/Brier/ECE/classwise ECE; entropy | correct vs wrong; held-out objective AUROC / reject score. Softmax over K known types is not “unknown.”

8. **Stochastic rollouts + distributional BC metrics**  
   Sample actions; store logits; report NLL/KL alongside top-1. Greedy argmax understates uncertainty.

9. **Guard BC conditioning provenance**  
   Typed conditioning with `ctx_used ≤ ctx`; support per-timestep `z_t` / `b_t`. Split seed ≠ init seed.

10. **Cross-play dataset + multi-predator path**  
    Shared prey / shared env seeds across predator objectives; record all `P` predators. Bridge to Part 2 without building planners yet.

## What must NOT be claimed yet

- SOTA or any external superiority
- “Encoder recovers strategy” without beating survival-time / random-encoder controls under held-out `ckpt_seed`
- “GRU-JEPA beats window-JEPA / VAE” until `evaluate_encoders` actually runs GRU with matched budgets
- “LayerNorm prevents collapse”
- “Posterior / Bayesian / uncertainty-aware” for current `belief.py` (soft classifier confidence only)
- Robustness to novel or adapting opponents (no OOD / switch protocol yet)
- Latent-conditioned BC improves opponent modeling (post-capture mass, unguarded `z`, greedy targets)
- Identity-JEPA numbers as self-supervised evidence

Honest current statement: *SSL + eval probe scaffold with episode-level (not checkpoint-level) splits; not yet a sequential uncertainty-aware opponent model.*

## Standard experiment matrix

Every published cell: ≥3–5 encoder seeds × ≥3–5 BC seeds, fixed episode+checkpoint splits, mean ± CI, JSON artifact.

| Axis | Levels |
|------|--------|
| Encoder (`lat ∈ {2,8,16/32}`) | GRU-JEPA (primary), window JEPA, β-VAE, random frozen, supervised oracle (same features), supervised contrastive (separate axis) |
| Split | Within-ckpt episode; leave-one-ckpt-out; leave-one-objective-out |
| Prefix `k` | `{5,10,25,50,100}` length-matched |
| Downstream | `π(a|s)`, `π(a|s,z_t)`, `Σ_k b_t π_k`, oracle `π(a|s,y)`, shortcut `π(a|s, survival_time)` |

**Metrics:** probe, oracle gap, GMM ARI (multi-seed), effective rank; NLL/Brier/ECE ± temperature; mixture-BC top-1 + NLL/KL; OOD AUROC + BC drop; (later) switch-episode tracking lag.

**Minimum bar before comparative claims:** held-out-checkpoint probe beats survival-time and random-encoder floors with non-overlapping CIs; collapse diagnostics stable; mixture-BC uses `b_t` not frozen full-episode `z`.

## Keep vs defer

**Keep minimal:** flat NumPy datasets, dataclasses, sklearn probe (eval), one GRU, simple BC, explicit grouped splits.

**Add before results count:** manifests, causal feature schema, lengths/masks, checkpoint splits, online encode API, belief→mixture BC, cross-play generation.

**Defer (Part 2):** planners, world models, co-training, continuous latent dynamics, sophisticated particle filters, full mixed-team generalization claims.
