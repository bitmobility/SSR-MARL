"""Mixed training tasks and initial-state distributions."""

formations = ("bar", "triangle", "square")
path_types = ("line", "left_turn", "sine")
time_limits = {"line": 120.0, "left_turn": 120.0, "sine": 180.0}
initial_lateral_range = (-0.2, 0.2)  # m, positive to the left
initial_heading_range = (-0.08726646259971647, 0.08726646259971647)  # rad
episode_seed_rule = (
    "SeedSequence(seed+env_id, spawn_key=(4,env_id,episode_index)); "
    "disturbance/initial_pose streams=2/3; balanced combination queue=make_rng(seed,7)"
)
