# Algorithms

All methods share actor parameters across vehicles. Each actor receives 21 features per vehicle and outputs two normalized controls.

| Method | Implementation |
|---|---|
| Ours | Tanh-Gaussian actor, set-based team critic with DOB features, safety-filter reward feedback |
| IPPO | Shared recurrent local actor/critic, complete-episode updates |
| MADDPG | Deterministic actor, centralized per-vehicle Q, target networks, uniform FIFO replay |
| MAPPO-Lag | Reward and cost critics, projected nonnegative safety multiplier |

PPO ratios use nominal policy actions; dynamics and DOB use executed actions. Vehicle masks exclude padding. Reward values bootstrap at time limits, but not at true termination. MAPPO-Lag's episodic safety cost does not bootstrap at time limits.

Construct agents with `training=False` for inference and load weights with `load_model()`. Reset IPPO's recurrent state at each episode.
