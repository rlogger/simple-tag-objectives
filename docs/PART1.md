# Part 1: Uncertainty-aware opponent modeling

Part 1 updates a **policy model of opponent behavior** that is robust to
opponents that adapt and use varying strategies. Part 2 (planner that samples
from this model) is explicitly out of scope for this scaffold.

## Problem

In the objective-typed predator env ([ENVIRONMENT.md](ENVIRONMENT.md)), the
predator’s hidden objective (`capture` / `risk` / `curious`) shapes behavior
the prey must infer online. Labels exist for **evaluation only**; the encoder
is trained self-supervised.

## Pipeline

```text
tag_objectives env
      │
      ▼
MAPPO specialists (scripts/train_mappo.py)
      │  capture / risk / curious checkpoints
      ▼
mopa.data.objective_dataset  →  labeled rollouts
      │
      ├─► GRU-JEPA (primary)     mopa.encoders.train_jepa_gru
      ├─► window JEPA / VAE      baselines
      ▼
latent z (point)  ──►  mopa.belief  ──►  soft posterior + entropy
      │
      ▼
latent-conditioned BC   π(a|s,z) vs π(a|s)   mopa.bc
```

## Strategy map

| Component | Choice | Why |
|-----------|--------|-----|
| Env instrument | Objective-typed predators | Strategy = reward objective, not just target positions |
| Specialist policies | MAPPO (CTDE MLP) | Stable labeled rollouts for clustering / BC |
| Primary encoder | GRU-JEPA | Variable-length “episode so far”; L2; LayerNorm; no VICReg |
| Baselines | Window JEPA, β-VAE | Ablations vs SSL reconstruction |
| Ceiling | Supervised MLP oracle | Probe–oracle gap |
| Uncertainty | Soft posterior + entropy | Belief, not point `z` alone |
| Opponent policy | Latent-conditioned BC | Downstream use of belief |
| Splits | Episode-level | No pooled-timestep leakage |

## Metric protocol (claim SOTA only after this)

Report **mean ± std over ≥3 encoder/BC seeds**, fixed episode splits.

1. **Representation**
   - Linear probe accuracy on latents
   - Supervised oracle accuracy (trajectory features → label)
   - Probe–oracle gap
   - GMM ARI (cluster recovery)
2. **Anytime**
   - Probe / ARI vs prefix length `k` (GRU prefixes)
3. **Uncertainty**
   - Mean posterior entropy vs correctness
   - ECE (`mopa.metrics.expected_calibration_error`)
4. **Downstream**
   - BC top-1: `π(a|s)` vs `π(a|s,z)`
   - Optional OOD: held-out registered objective

## Non-goals (deferred Part 2)

- MuZero / TD-MPC / MA-TDMPC planners
- CPL with planner counterfactuals
- Full co-training loop (blue plans with red model updating online)

## Quick commands

```bash
# train specialists
python scripts/train_mappo.py alg=mappo_objectives_capture NUM_SEEDS=3
python scripts/train_mappo.py alg=mappo_objectives_risk NUM_SEEDS=3
python scripts/train_mappo.py alg=mappo_objectives_curious NUM_SEEDS=3

# encode + believe (sketch)
from mopa import train_jepa_gru, fit_latent_belief, metrics
# z = train_jepa_gru(X, lengths, key); probe, ari = metrics(z_full, y, 3)
# belief = fit_latent_belief(z_train, y_train, z_val)
```
