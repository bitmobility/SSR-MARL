# Entry Points

Run scripts from the repository root. See the [project README](../README.md) for commands.

| Script | Purpose |
|---|---|
| `train_proposed.py` | Ours |
| `train_ippo.py`, `train_maddpg.py`, `train_mappo_lag.py` | Baselines |
| `train_ablation.py` | Safety-filter, DOB and coordination ablations |
| `test_post_trained_agent.py` | Main performance comparison |
| `test_ablation.py` | Ablation evaluation |
| `test_generalization.py` | Vehicle-count and payload-shape generalization |
| `test_robust.py` | Model-error robustness |
| `evaluate.py` | A trained policy on one configuration |

Set `MODEL_PATH` before running paper evaluations. New outputs are stored in `runs/`.
