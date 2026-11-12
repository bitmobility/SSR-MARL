from pathlib import Path as _Path

seeds = [11,12,13,14,15]
render = False
num_envs = 8
total_steps = 1_200_000
rollout_steps = 128
cbf_reward_weight = 0.03
cbf_reward_sigma = 0.4
outcome_window = 50
output_dir = str(_Path(__file__).resolve().parents[1] / "runs")
experiment_name = 'train_proposed'
swanlab_enabled = False
swanlab_project = 'SRMARL'
