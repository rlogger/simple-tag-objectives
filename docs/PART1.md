# Part 1: Uncertainty-aware opponent modeling

Part 1 updates a **policy model of opponent behavior** that is robust to
opponents that adapt and use varying strategies. Part 2 (planner that samples
from this model) is explicitly out of scope for this scaffold.

Foundation hardening details and claim bar: [PART1_GUIDELINES.md](PART1_GUIDELINES.md).

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
mopa.data.objective_dataset  →  labeled rollouts (+ capture-aware lengths)
      │
      ├─► GRU-JEPA (primary)     train_jepa_gru_with_params / encode_jepa_gru
      ├─► window JEPA / VAE      baselines
      ▼
latent z_t (prefix)  ──►  eval probe (fit_latent_belief)  ──►  soft scores
      │
      ▼
point-latent BC baseline   π(a|s,z) vs π(a|s)   mopa.bc
      │
      ▼
scripts/run_part1.py  →  JSON manifest
```

The logistic probe is an **evaluation instrument**, not the training-time belief
for the action model. Belief-mixture BC (`Σ_k b_t(k) π_k`) is still future work.

## Strategy map

| Component | Choice | Why |
|-----------|--------|-----|
| Env instrument | Objective-typed predators | Strategy = reward objective, not just target positions |
| Specialist policies | MAPPO (CTDE MLP) | Stable labeled rollouts for clustering / BC |
| Primary encoder | GRU-JEPA + unit-sphere | Variable-length prefixes; L2-on-normalized; pad-masked; no VICReg |
| Baselines | Window JEPA, β-VAE | Ablations vs SSL reconstruction |
| Ceiling | Supervised MLP oracle | Probe–oracle gap |
| Eval uncertainty | Soft probe scores + entropy / ECE | Labels for eval only |
| Opponent policy (current) | Point-latent BC baseline | Downstream sanity check |
| Splits | Checkpoint-grouped (primary) | Held-out opponent instance |

## Metric protocol (do not claim SOTA yet)

Report **mean ± std over ≥3 encoder/BC seeds**, **checkpoint-held-out** splits,
train-only encode / probe / GMM fit.

1. **Representation**
   - Held-out linear probe accuracy
   - Survival-time probe **control**
   - GMM ARI on held-out latents
   - Collapse diagnostics (across-episode std, effective rank)
2. **Anytime**
   - Probe / ARI vs prefix length `k` (length-matched)
3. **Uncertainty (eval probe)**
   - ECE on held-out soft scores
4. **Downstream baseline**
   - BC top-1: `π(a|s)` vs `π(a|s,z)` (point latent; not belief mixture)

## Non-goals (deferred)

- Belief-mixture BC / sequential Bayes filter
- MuZero / TD-MPC planners and co-training
- Cross-play prey holdout / open-set reject (roadmap items 5–10)

## Quick commands

```bash
# train specialists
python scripts/train_mappo.py alg=mappo_objectives_capture NUM_SEEDS=3
python scripts/train_mappo.py alg=mappo_objectives_risk NUM_SEEDS=3
python scripts/train_mappo.py alg=mappo_objectives_curious NUM_SEEDS=3

# validate checkpoints / write stub manifest
python scripts/run_part1.py --dry-run-checks --out artifacts/part1_dry.json

# smoke Part 1 run (needs checkpoints under logs/)
python scripts/run_part1.py \
  --n-eps 16 --encoder-steps 200 --bc-steps 200 \
  --split checkpoint --out artifacts/part1_run.json
```
