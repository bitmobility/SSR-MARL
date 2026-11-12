from pathlib import Path as _Path

seeds = [11,12,13,14,15]
render = False
num_envs = 8
total_steps = 1_200_000
warmup_steps = 10000
update_every = 50
gradient_steps = 1
exploration_std_start = 0.2
exploration_std_end = 0.05
exploration_decay_steps = 600000
outcome_window = 50
output_dir = str(_Path(__file__).resolve().parents[1] / "runs")
experiment_name = 'train_maddpg'
swanlab_enabled = False
swanlab_project = 'SRMARL'
