# Configuration

| Location | Settings |
|---|---|
| `train_*.py` | Seeds, total steps, parallel environments and outputs |
| `train_ablation.py` | Ablation variants and safety overrides |
| `alg_cfgs/` | Networks, optimization and algorithm parameters |
| `env_cfgs/transport.py` | Dynamics, rewards and termination limits |
| `env_cfgs/sampling.py` | Mixed tasks, time limits and initial poses |
| `env_cfgs/reference.py` | Path geometry and preview |
| `env_cfgs/disturbance.py` | Vehicle model errors |
| `env_cfgs/safety.py` | DOB, HOCBF and active coordination |

Each seed creates independent networks, optimizers, environments and buffers. Evaluation reads the checkpoint's configuration snapshot. Test model lists and scenarios are defined at the top of `scripts/test_*.py`.
