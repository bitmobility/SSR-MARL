# Safety Filter

`SafetyFilter` accepts and returns physical actions `[acceleration, steering]` with shape `(N, 2)`. Its joint QP uses `[acceleration, sin(slip_angle)]`. Parameters are in [safety.py](../../config/env_cfgs/safety.py).

```python
obs, info = env.reset()
safety.reset(env)
state = env.get_state()
action, diagnostics = safety.filter(state, env.scale_action(policy_action))
obs, reward, terminated, truncated, info = env.step(action)
if info["valid_transition"]:
    safety.observer.update(
        state, info["executed_action"], info["full_state"], info["elapsed_time"])
```

| Module | Purpose |
|---|---|
| `nominal_model.py` | Input transforms and control/speed bounds |
| `disturbance_observer.py` | Acceleration and yaw-rate error estimates |
| `coordination_cbf.py` | Coordination derivatives and active objective |
| `robust_hocbf.py` | DOB compensation and residual-error margins |
| `safety_filter.py` | QP solve, action recovery and diagnostics |

Reset the filter after each environment reset. By default, QP failure returns the nominal action and sets `fallback_used=True`; `fallback=False` raises an error. `correction_cost` is the mean squared normalized action correction before reward weighting.
