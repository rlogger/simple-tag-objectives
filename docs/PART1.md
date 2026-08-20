# Part 1: uncertainty-aware opponent modeling

Part 1 models an objective-typed predator online. Strategy labels exist for
evaluation and supervised policy fitting, but never enter the self-supervised
encoder or online inference API.

The later update deck explicitly removed MuZero/TD-MPC planners for now. This
repository therefore implements the opponent-modeling stack and a clean CPL
integration boundary, without presenting a placeholder planner as completed
model-based co-training.

## Data flow

```text
objective-typed environment
        |
        v
three MAPPO predator specialists + one fixed prey checkpoint family
        |
        v
capture-aware trajectories with checkpoint/reset/length provenance
        |
        +--> GRU-JEPA (primary variable-length representation)
        +--> fixed-window JEPA / beta-VAE / random controls
        +--> supervised trajectory oracle
        |
        v
checkpoint-held-out probe, 3-GMM ARI, collapse, anytime, calibration
        |
        +--> causal point-z BC: pi(a_t | s_t, z_{t-1})
        |
        +--> per-strategy policies + Bayesian filter
                    |
                    v
             sum_k b_t(k) pi_k(a_t | s_t)
```

`scripts/run_part1.py` executes this flow and writes one schema-v2 JSON
manifest.

## Leakage controls

- The primary split holds out whole MAPPO checkpoint seeds and is reused across
  encoder and BC initialization seeds.
- All scaling, encoders, probes, GMMs, policy heads, and calibration fits use
  the training fold only. Held-out examples are encoded with frozen parameters.
- One fixed prey checkpoint family is used across capture, risk, and curious
  predators. `--prey-objective matched` remains available only as an explicitly
  confounded smoke/control setting and is rejected for full runs.
- Each episode records its JAX reset key, checkpoint seed, strategy label, fold,
  and valid first-capture length. Reset keys are matched across predator
  objectives for each checkpoint/episode index.
- Predator policy state includes all agent positions, a velocity proxy,
  predator identity, and the observed lava geometry. Resources are intentionally
  absent because predators do not observe them.
- Action `a_t` is conditioned only on the GRU prefix through `t-1`, whose final
  transition is already available before `a_t`. A full-episode latent is never
  attached to an earlier action.

## Representation protocol

GRU-JEPA uses a pad-safe GRU, LayerNorm, unit-sphere embeddings, an EMA target,
and squared L2 prediction loss. Length-one episodes remain encodable but are
excluded from JEPA updates because they have no strictly later target. The
implementation does not use BatchNorm, L1 prediction, or VICReg.

The matched headline comparison uses the same declared context prefix for
GRU-JEPA, fixed-window JEPA, beta-VAE, random projection, and the supervised
oracle. Fixed-window JEPA trains only on episodes with a complete future target;
padded targets are never treated as real trajectories. Anytime curves use the
latest available fixed window, while the GRU can continue using longer prefixes.
The manifest records each model's effective prefix so a saturated fixed-window
baseline cannot be mistaken for a length-matched long-context model.

## Online belief and action models

`mopa.strategy` fits one discrete action policy per strategy on training labels,
then removes labels from inference. A Bayesian switching filter updates the
strategy posterior from observed action likelihoods. The action at time `t` is
scored with the predictive belief before that action is observed:

```text
p(a_t | s_t, history) = sum_k b_t(k) pi_k(a_t | s_t)
```

The manifest reports mixture top-1/NLL, the true-strategy policy ceiling,
predictive and posterior strategy accuracy, entropy, NLL, Brier score, ECE,
classwise ECE, and reliability-bin data. Declared action classes receive floor
probability even if a class is absent from the training fold.

The separate neural BC comparison reports both `pi(a|s)` and causal
`pi(a|s,z)` accuracy/NLL on the exact same validation mask.

## CPL boundary

`mopa.replay` defines immutable, padded trajectories with strict shapes, masks,
and planner checkpoint/version provenance. `mopa.cpl` provides preference-pair
integrity checks, a differentiable Bradley-Terry CPL loss, and callback-driven
counterfactual suffix generation that:

1. preserves the original prefix;
2. resamples the opponent action;
3. asks the blue planner to replan every suffix step; and
4. rejects any planner checkpoint or version drift.

A real preference dataset is not present because the current scope has no
trained planner. CPL stages therefore remain `not_run`; unit-tested API
readiness is not reported as an experiment.

## Required outputs

The manifest includes:

- environment behavior by objective: capture rate, survival time, resources,
  predator/prey lava steps, and predator coverage;
- held-out probe, three-component GMM ARI, random/survival/oracle controls, and
  collapse diagnostics;
- prefix/anytime and uncertainty reports;
- unconditioned, causal point-latent, belief-mixture, and oracle-policy metrics;
- exact config, dependency versions, commit/dirty state, checkpoint hashes,
  split membership, reset keys, valid lengths, dataset hash when cached, and
  per-stage execution status.

## Run semantics

- `dry_run` validates inputs and writes no experiment claim.
- `smoke_passed` means a tiny or synthetic code path executed.
- `full_run_finished` means the declared clean, multi-seed computation finished.
  It does not say the budgets were adequate or a scientific threshold was met.

The source documents leave full budgets, confidence intervals, adaptive switch
schedules, CPL preference ordering, and success thresholds open. Record those
choices before running and apply the claim bar in
[PART1_GUIDELINES.md](PART1_GUIDELINES.md). The current implementation/deferred
map is in [STATUS.md](STATUS.md).
