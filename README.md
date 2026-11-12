# Safe MARL for Cooperative Transportation

Ours combines shared-policy PPO, a disturbance observer, robust high-order CBFs, active coordination, and safety-filter correction feedback. The repository also includes IPPO, MADDPG, MAPPO-Lag, and the paper's ablation and evaluation scripts.

## Setup

Python 3.10+:

```bash
pip install -r requirements.txt
```

The requirements include `pygame` for the optional 2D viewer. Install `swanlab` for online logging. Local CSV logging is enabled by default.

## Training

Run from this directory:

```bash
python scripts/train_proposed.py
python scripts/train_ippo.py
python scripts/train_maddpg.py
python scripts/train_mappo_lag.py
python scripts/train_ablation.py
```

Training uses a single budget of 1,200,000 environment steps. Each shuffled cycle covers all nine formation/path combinations at model error scale 1. Active episodes finish after the budget is reached.

Edit `config/train_*.py` for seeds, `total_steps`, and sampling budgets; `config/alg_cfgs/` for algorithm parameters; and `config/env_cfgs/sampling.py` for tasks and initial-state distributions. Ablations support `--variant no_safety`, `nominal_hocbf`, `no_coordination`, or `full`.

Each run writes `config.yaml`, `logs/episodes.csv`, and `checkpoints/actor.pt` under `runs/`.

## Pretrained Example

We provide a pretrained model of our method (Ours) for demonstration, together with its training results in [`train_proposed/`](train_proposed/). Five runs (seeds 11-15) are included; seed13 is used in the example below.

```bash
python scripts/evaluate.py --method Ours --checkpoint train_proposed/seed13/checkpoints/actor.pt --formation triangle --path sine
```

Add `--render` to view the rollout. Each run contains the actor weights, a loadable `config.yaml`, episode logs, and the original SwanLab records. See the [model notes](train_proposed/README.md).

## Evaluation

Ours evaluation paths point to the included seed13-seed15 models. Set `MODEL_PATH` for baseline and ablation checkpoints before evaluating those methods.

```bash
python scripts/test_post_trained_agent.py
python scripts/test_ablation.py
python scripts/test_generalization.py
python scripts/test_robust.py
```

Use `--episodes_per_combination 1 --model_limit 1` for a small run, or `--help` for options. Results include episode records, CSV/Excel tables, cases, and a manifest. Set `--output_dir` to change the destination.

For a single configuration:

```bash
python scripts/evaluate.py --method Ours --checkpoint /path/to/run/checkpoints/actor.pt --formation triangle --path sine
```

Methods: `Ours`, `IPPO`, `MADDPG`, `MAPPO-Lag`. Paths: `line`, `left_turn`, `sine`, `arc`, `z`, `omega`. Add `--render` for the viewer or `--output runs/evaluation.csv` to save metrics. Existing trained checkpoints remain supported.

## Layout

```text
config/       Algorithm, training, task and safety parameters
scripts/      Training, ablation and evaluation entry points
src/alg/      Ours and three baselines
src/buffer/   Rollout and replay buffers
src/cbf/      DOB, robust HOCBF and safety filter
src/env/      Dynamics, geometry, tasks and parallel sampling
src/utils/    Seeding, logging and evaluation helpers
train_proposed/  Pretrained Ours models and example training results
```
