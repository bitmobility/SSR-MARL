# Environment

`TransportEnv` handles dynamics, rigid payload fitting, observations, rewards and termination. `TrainingEnv` samples reproducible initial conditions. Parallel pools draw balanced formation/path combinations from a shared queue.

- Observations: `(N, 21)`; physical actions: `(N, 2)` acceleration and steering.
- Convert policy outputs with `env.scale_action()` before execution.
- `reset()` returns `(obs, info)`; `step()` returns `(obs, reward, terminated, truncated, info)`.
- Update the DOB only after valid transitions, using the executed action and elapsed time.
- Collision, coordination error, path deviation and degenerate fitting are checked at each physics substep.
- Time limits are truncations; bootstrap from the state before reset.

Workers use `spawn` and process only real vehicles. Padding and vehicle masks remain in the main process. The no-safety ablation bypasses action correction while retaining DOB features for the critic.
