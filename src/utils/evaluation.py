from Code.config import parameters as copy_config

import numpy as np

import Code.config.env_cfgs.disturbance as disturbance_settings
from Code.src.env import Disturbance, FORMATIONS
from Code.src.utils.random import make_rng


TRAIN_PATH_TYPES = ("line", "left_turn", "sine")
GENERALIZATION_PATH_TYPES = ("arc", "omega", "z")

PATH_TYPES = TRAIN_PATH_TYPES + GENERALIZATION_PATH_TYPES


def build_cases(config, model_config, conditions):
    if config.episodes_per_combination < 1:
        raise ValueError("Each test combination needs at least one episode")
    if not 0 <= config.disturbance_scale <= 1:
        raise ValueError("Model error scale must be in [0, 1]")
    if config.generalization_time_limit <= 0:
        raise ValueError("Generalization time limit must be positive")
    if not conditions["scenarios"]:
        raise ValueError("At least one test scenario is required")
    seen = set()
    for scenario in conditions["scenarios"]:
        if not scenario["group"] or not scenario["formations"] or not scenario["path_types"]:
            raise ValueError("Each scenario needs a group, formations and paths")
        if any(name not in FORMATIONS for name in scenario["formations"]):
            raise ValueError("Test formations must be defined in FORMATIONS")
        if any(path not in PATH_TYPES for path in scenario["path_types"]):
            raise ValueError("Choose line, left_turn, sine, arc, omega or z")
        for formation in scenario["formations"]:
            for path in scenario["path_types"]:
                key = (scenario["group"], formation, path)
                if key in seen:
                    raise ValueError(f"Duplicate test scenario: {key}")
                seen.add(key)
    enabled = model_config["env"]["disturbance_enabled"]
    scale = config.disturbance_scale if enabled else 0.0
    disturbance = Disturbance(enabled, 6, make_rng(config.seed, 5),
                              copy_config(disturbance_settings, **model_config["disturbance"]))
    cases = []
    for scenario in conditions["scenarios"]:
        group = scenario["group"]
        for formation in scenario["formations"]:
            n = len(FORMATIONS[formation][0])
            for path_type in scenario["path_types"]:
                for repeat in range(config.episodes_per_combination):
                    seed = int(np.random.SeedSequence(config.seed, spawn_key=(
                        5, PATH_TYPES.index(path_type), repeat)).generate_state(1)[0])
                    parameters = disturbance.sample(6, rng=make_rng(seed, 0), scale=scale)[:n]
                    cases.append(dict(group=group, formation=formation, n_agents=n, path_type=path_type,
                                      case_id=f"{group}_{formation}_{path_type}_{repeat:03d}", repeat_index=repeat,
                                      episode_seed=seed, disturbance_scale=scale,
                                      disturbance_parameters=parameters.tolist()))
    return cases


def append_trajectory(states, actions, info, cbf=None):
    state = info["full_state"]
    frame = {key: np.array(state[key], copy=True) for key in (
        "time", "payload_position", "payload_heading", "vehicle_positions", "vehicle_headings", "vehicle_speeds")}
    frame.update({key: np.array(info[key], copy=True) for key in (
        "attachment_norms", "path_error", "heading_error")})
    states.append(frame)
    if cbf is not None:
        actions.append(dict(nominal_action=cbf["nominal_action"].copy(),
                            executed_action=info["executed_action"].copy(),
                            cbf_correction_cost=cbf["correction_cost"]))


def pack_trajectory(states, actions):
    data = {key: np.stack([frame[key] for frame in states]) for key in states[0]}
    data.update({key: np.stack([step[key] for step in actions]) for key in actions[0]})
    return data


def sampling_config(snapshot):
    """Read task distributions from current or legacy model snapshots."""
    return snapshot["sampling"] if "sampling" in snapshot else snapshot["curriculum"]
