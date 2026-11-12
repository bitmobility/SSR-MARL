from pathlib import Path as _Path

seeds = [11, 12, 13, 14, 15]
render = False
num_envs = 8
total_steps = 1_200_000
rollout_steps = 128
cbf_reward_weight = 0.03
cbf_reward_sigma = 0.4
outcome_window = 50
output_dir = str(_Path(__file__).resolve().parents[1] / "runs")
experiment_name = "train_ablation"
swanlab_enabled = False
swanlab_project = "SRMARL"

default_variants = ("no_safety", "nominal_hocbf", "no_coordination")
variants = {
    "no_safety": dict(use_safety_layer=False, safety=dict(robust=False, coordination_weight=0.0)),
    "nominal_hocbf": dict(use_safety_layer=True,
                         safety=dict(robust=False, nominal_coordination_gains=(4.0, 4.0))),
    "no_coordination": dict(use_safety_layer=True, safety=dict(coordination_weight=0.0)),
    "full": dict(use_safety_layer=True, safety={}),
}
