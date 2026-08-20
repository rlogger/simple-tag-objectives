# Experiment and claim guidelines

See [STATUS.md](STATUS.md) for the requirement-to-code map. These rules keep a
runnable scaffold, a smoke check, and a scientific result distinct.

## Data and split rules

- Freeze each episode at first capture and exclude padded/post-capture actions.
- Hold out whole checkpoint seeds by default. Fit feature scaling, encoders,
  probes, GMMs, policy models, and calibration only from the declared train
  fold.
- Reuse one manifest split for every encoder and policy initialization seed.
- Use one fixed prey checkpoint family across predator objectives. A matched
  co-trained prey is a confounded control, not a full strategy experiment.
- Record objective label, checkpoint seed, environment seed, valid length, and
  action timestep explicitly.
- Match environment reset keys across objective labels so layout randomness is
  not a strategy shortcut.
- Use the same per-step trajectory feature schema and held-out examples for
  GRU-JEPA, window JEPA, beta-VAE, random, and oracle comparisons.
- Give fixed-window JEPA only real future targets; never optimize against a
  wholly or partly padded target as if it were observed.

## Causality and label rules

- SSL encoders never receive objective labels.
- A linear probe may use labels as an evaluation instrument.
- Per-strategy action heads may use training labels; online inference may not.
- Predict action `a_t` from the causal `z_{t-1}` prefix or the predictive belief
  based on actions before `t`. A full-episode latent is never a valid
  conditioning input for an earlier action.
- Counterfactual pairs must preserve the original prefix and planner
  checkpoint/version. Reject provenance drift.

## Required metrics

- Representation: held-out probe accuracy, three-component GMM ARI, random and
  survival-time controls, oracle gap, collapse diagnostics, and prefix curves.
- Belief: posterior entropy, NLL, Brier score, ECE, classwise ECE, and
  reliability-bin data.
- Policy: top-1 action accuracy and action NLL for unconditioned, point-latent,
  belief-mixture, and oracle variants where applicable.
- Environment: capture rate, survival time, resources collected, predator and
  prey lava steps, and predator spatial coverage. Do not compare raw rewards
  across different predator objectives.

## Run labels

- `dry_run`: validates declared inputs only.
- `smoke`: exercises code paths with tiny or synthetic data; stages may be
  `smoke_passed`, never `full_run_finished`.
- `full`: requires clean, fixed-prey, fresh-rollout, checkpoint-held-out,
  three-seed execution. Finished stages use `full_run_finished`, which records
  execution rather than scientific adequacy.

Do not claim strategy recovery, robustness to adapting opponents, planner
benefit, or algorithm superiority until the corresponding full manifests and
artifacts exist.
