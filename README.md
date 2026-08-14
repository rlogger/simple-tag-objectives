# simple-tag-objectives

Minimal JaxMARL environment: objective-typed predators vs one prey, with
resources and lava. Extracted and cleaned from
[marl-opp-aware](https://github.com/rlogger/marl-opp-aware) — env + scaffold
only, no trainers, checkpoints, or paper artifacts.

See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the contract.

## Install

JaxMARL is required and is not on PyPI as a first-class dep here — install it
editable from a local checkout (or your preferred source):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e /path/to/JaxMARL
pip install -e ".[dev]"
```

## Quick start

```python
import jax
from tag_objectives import make_env, evaluate_policy, random_policy

env = make_env("curious")  # or "capture" / "risk" / any registered name
metrics = evaluate_policy(
    env, random_policy(env), n_eps=64, key=jax.random.PRNGKey(0)
)
print(metrics.means())
```

## Layout

```text
src/tag_objectives/
  resources.py    # SimpleTagResourcesMPE
  objectives.py   # SimpleTagObjectivesMPE + objective registry
  api.py          # make_env / evaluate_policy / random_policy
  types.py        # EpisodeMetrics
  teams.py        # freeze helpers for batched eval
tests/            # golden rewards + API contract
docs/ENVIRONMENT.md
```

## Test

```bash
pytest -q
```
