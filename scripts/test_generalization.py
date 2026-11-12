"""Evaluate unseen vehicle counts and payload configurations."""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

CODE_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    from importlib.machinery import ModuleSpec
    from importlib.util import module_from_spec
    spec = ModuleSpec("Code", loader=None, is_package=True)
    spec.submodule_search_locations = [str(CODE_ROOT)]
    sys.modules["Code"] = module_from_spec(spec)

import numpy as np
from openpyxl import Workbook
import yaml

from Code.src.utils.evaluation import sampling_config
from Code.config import parameters
import Code.config.env_cfgs.transport as env_settings
import Code.config.env_cfgs.reference as reference_settings
import Code.config.env_cfgs.disturbance as disturbance_settings
import Code.config.env_cfgs.safety as safety_settings
from Code.src.alg.proposed import Proposed
from Code.src.alg.baselines.ippo import IPPO
from Code.src.alg.baselines.maddpg import MADDPG
from Code.src.alg.baselines.mappo_lag import MAPPOLag
from Code.src.cbf import SafetyFilter
from Code.src.env import FORMATIONS, Observation, TransportEnv
from Code.src.utils.evaluation import GENERALIZATION_PATH_TYPES, build_cases
from Code.src.utils.metrics import TestEpisodeStats
from Code.src.utils.random import make_rng, set_seed
from Code.src.utils.test_manifest import build_test_manifest


MODEL_PATH = {
    "IPPO": [


        "experiments/train_ippo/mixed_seed11_20261022_232340_871119/checkpoints/actor.pt",
        "experiments/train_ippo/mixed_seed12_20261023_011124_000281/checkpoints/actor.pt",
        "experiments/train_ippo/mixed_seed15_20261023_053535_992699/checkpoints/actor.pt",
    ],

    "MADDPG": [

        "experiments/train_maddpg/mixed_seed11_20261022_232425_900491/checkpoints/actor.pt",
        "experiments/train_maddpg/mixed_seed12_20261023_002807_228426/checkpoints/actor.pt",
        "experiments/train_maddpg/mixed_seed15_20261023_033922_544465/checkpoints/actor.pt",
    ],

    "MAPPO-Lag": [

        "experiments/train_mappo_lag/mixed_seed11_20261022_232544_861974/checkpoints/actor.pt",
        "experiments/train_mappo_lag/mixed_seed12_20261023_004217_523722/checkpoints/actor.pt",
        "experiments/train_mappo_lag/mixed_seed15_20261023_043221_911827/checkpoints/actor.pt",
    ],

    "Ours": [

         "train_proposed/seed13/checkpoints/actor.pt",
         "train_proposed/seed14/checkpoints/actor.pt",
         "train_proposed/seed15/checkpoints/actor.pt",

    ],
}

AGENT_MAP = {"IPPO": IPPO, "MADDPG": MADDPG, "MAPPO-Lag": MAPPOLag, "Ours": Proposed}


ACTION_DETERMINISTIC = {"IPPO": False, "MADDPG": True, "MAPPO-Lag": False, "Ours": True}


TEST_CONDITIONS = dict(scenarios=(
    dict(group="rectangle", formations=("rectangle_4", "rectangle_5", "rectangle_6"),
         path_types=("arc", "omega", "z")),
    dict(group="shape", formations=("pentagon_5", "t_3", "l_3"),
         path_types=("arc", "omega", "z")),
))

TABLE_GROUP_BY = {"rectangle": "formation", "shape": "formation"}
TEST_SETTINGS = {
    "episodes_per_combination": 10,
    "seed": 2027,
    "disturbance_scale": 1.0,
    "generalization_time_limit": 180.0,
    "device": "cpu",
    "render": False,
}
OUTPUT_DIR = CODE_ROOT / "runs" / "test_generalization"


TABLE_METRICS = (
    ("return", "env_return", 1.0),
    ("success_rate", "success", 100.0),
    ("violation_rate", "coordination_violation", 100.0),
    ("connection_error", "attachment_mean", 1.0),
    ("tracking_rmse", "path_rmse", 1.0),
)


def read_models(agent_name, model_limit=None):
    methods = list(MODEL_PATH) if agent_name == "all" else [agent_name]
    models = []
    for method in methods:
        paths = MODEL_PATH[method][:model_limit]
        if not paths:
            raise ValueError(f"{method}: Set checkpoint paths in MODEL_PATH")
        seeds = set()
        for path in paths:
            checkpoint = (CODE_ROOT / path).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text(encoding="utf-8"))
            algorithm = AGENT_MAP[method].algorithm
            if algorithm not in config:
                raise ValueError(f"{checkpoint}: Missing {algorithm} network configuration")
            seed = int(config["train"]["seed"])
            if seed in seeds:
                raise ValueError(f"{method}: Duplicate training seed: {seed}")
            seeds.add(seed)
            models.append(dict(method=method, seed=seed, checkpoint=checkpoint, config=config))
    return models


def make_test_cases(config, model_config):
    if not model_config["env"]["disturbance_enabled"]:
        raise ValueError("Generalization testing requires model errors to be enabled")
    cases = build_cases(config, model_config, TEST_CONDITIONS)
    sampling = sampling_config(model_config)
    for case in cases:
        case["scenario"] = case[TABLE_GROUP_BY[case["group"]]]
        case["time_limit"] = float(config.generalization_time_limit if case["path_type"] in GENERALIZATION_PATH_TYPES
                                   else sampling["time_limits"][case["path_type"]])
        if not np.isfinite(case["time_limit"]) or case["time_limit"] <= 0:
            raise ValueError(f"{case['path_type']}: Episode time limit must be finite and positive")
        rng = make_rng(case["episode_seed"], 3)
        case["initial_lateral_offset"] = float(rng.uniform(*sampling["initial_lateral_range"]))
        case["initial_heading_offset"] = float(rng.uniform(*sampling["initial_heading_range"]))
    return cases


def select_agent_action(agent, obs, hidden=None):
    method = next(name for name, cls in AGENT_MAP.items() if cls.algorithm == agent.algorithm)
    deterministic = ACTION_DETERMINISTIC[method]
    if agent.algorithm == "ippo":
        return agent.choose_action(obs, hidden=hidden, deterministic=deterministic)
    if agent.algorithm == "maddpg":
        return agent.choose_action(obs), None
    return agent.choose_action(obs, deterministic=deterministic), None


def run_episode(agent, env, safety, case):
    obs, info = env.reset(**{key: case[key] for key in (
        "formation", "path_type", "disturbance_parameters",
        "initial_lateral_offset", "initial_heading_offset")})
    if safety is not None:
        safety.reset(env)
    hidden = None
    stats = TestEpisodeStats()
    phase = f"{agent.algorithm} / {case['case_id']}"
    env.render(phase=phase, info=info)
    while True:
        action, hidden = select_agent_action(agent, obs, hidden)
        nominal = env.scale_action(action)
        state = env.get_state()
        if safety is not None:
            executed, cbf = safety.filter(state, nominal)
        else:
            executed = nominal
            cbf = dict(correction_cost=0.0, fallback_used=False, status="disabled")
        obs, reward, terminated, truncated, info = env.step(executed, policy_action=action)
        if safety is not None and info["valid_transition"]:
            safety.observer.update(state, info["executed_action"], info["full_state"], info["elapsed_time"])

        stats.update(reward, 0.0, info, cbf)
        env.render(phase=phase, info=info)
        if env.render_closed:
            raise KeyboardInterrupt("Viewer closed")
        if terminated or truncated:
            break
    return stats.result()


def summarize_results(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ("method", "group", "scenario", "seed", "formation", "path_type"))
        groups.setdefault(key, []).append(row)
    per_combination = []
    for key, episodes in groups.items():
        result = dict(zip(("method", "group", "scenario", "seed", "formation", "path_type"), key))
        result["n_episodes"] = len(episodes)
        for name, source, scale in TABLE_METRICS:
            result[name] = float(np.mean([row[source] for row in episodes])) * scale
        per_combination.append(result)
    per_seed = []
    seed_fields = ("method", "group", "scenario", "seed")
    for key in dict.fromkeys(tuple(row[field] for field in seed_fields) for row in per_combination):
        combinations = [row for row in per_combination
                        if tuple(row[field] for field in seed_fields) == key]
        result = dict(zip(seed_fields, key))
        result.update(n_combinations=len(combinations),
                      n_episodes=sum(row["n_episodes"] for row in combinations))
        result.update({name: float(np.mean([row[name] for row in combinations]))
                       for name, _, _ in TABLE_METRICS})
        per_seed.append(result)
    summary = []
    table_fields = ("method", "group", "scenario")
    for key in dict.fromkeys(tuple(row[field] for field in table_fields) for row in per_seed):
        seeds = [row for row in per_seed if tuple(row[field] for field in table_fields) == key]
        result = dict(zip(table_fields, key))
        result["n_seeds"] = len(seeds)
        for name, _, _ in TABLE_METRICS:
            values = [row[name] for row in seeds]
            result[f"{name}_mean"] = float(np.mean(values))
            result[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(result)
    return per_combination, per_seed, summary


class TestLogger:
    def __init__(self, output_dir, config, models, cases):
        self.run_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.seed_counts = {method: sum(m["method"] == method for m in models)
                            for method in dict.fromkeys(m["method"] for m in models)}
        self.manifest = build_test_manifest(config, TEST_CONDITIONS, models, cases, ACTION_DETERMINISTIC)

        self.manifest["environment"]["sampling"]["time_limits"] = {
            case["path_type"]: case["time_limit"] for case in cases}
        self.manifest.update(
            table_group_by=TABLE_GROUP_BY,
            formations={name: dict(anchors=FORMATIONS[name][0].tolist(), vertices=FORMATIONS[name][1].tolist())
                        for name in dict.fromkeys(case["formation"] for case in cases)},
        )
        self.write_manifest()
        (self.run_dir / "cases.yaml").write_text(yaml.safe_dump(cases, sort_keys=False), encoding="utf-8")
        self.stream = (self.run_dir / "episodes.csv").open("w", encoding="utf-8-sig", newline="")
        self.writer = None

    def write_manifest(self):
        (self.run_dir / "manifest.yaml").write_text(
            yaml.safe_dump(self.manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def log_episode(self, model, case, metrics):
        method, seed = model["method"], model["seed"]
        row = dict(method=method, seed=seed, case_id=case["case_id"],
                   group=case["group"], scenario=case["scenario"], time_limit=case["time_limit"],
                   formation=case["formation"], n_agents=case["n_agents"], path_type=case["path_type"],
                   repeat_index=case["repeat_index"], episode_seed=case["episode_seed"],
                   disturbance_scale=case["disturbance_scale"],
                   initial_lateral_offset=case["initial_lateral_offset"],
                   initial_heading_offset=case["initial_heading_offset"],
                   **metrics)
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()
        self.rows.append(row)
        return row

    def save(self):
        if len(self.rows) != self.manifest["planned_episodes"]:
            raise ValueError("Incomplete test: episode data saved; summary skipped")
        per_combination, per_seed, summary = summarize_results(self.rows)
        workbook = Workbook()
        workbook.remove(workbook.active)
        for name, rows in (("table2", summary), ("per_seed", per_seed), ("per_combination", per_combination)):
            fields = list(rows[0])
            with (self.run_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            sheet = workbook.create_sheet(name)
            sheet.append(fields)
            for row in rows:
                sheet.append([row[field] for field in fields])
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
        workbook.save(self.run_dir / "generalization_results.xlsx")
        self.manifest.update(complete=True, completed_episodes=len(self.rows))
        self.write_manifest()

    def close(self):
        self.stream.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_name", choices=["all", *MODEL_PATH], default="all")
    parser.add_argument("--episodes_per_combination", type=int, default=TEST_SETTINGS["episodes_per_combination"])
    parser.add_argument("--seed", type=int, default=TEST_SETTINGS["seed"])
    parser.add_argument("--device", default=TEST_SETTINGS["device"])
    parser.add_argument("--render", action="store_true", default=TEST_SETTINGS["render"])
    parser.add_argument("--model_limit", type=int, default=None, help="Test only the first N models per method")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    config = SimpleNamespace(**dict(TEST_SETTINGS, episodes_per_combination=args.episodes_per_combination,
                                    seed=args.seed, device=args.device, render=args.render))

    models = read_models(args.agent_name, args.model_limit)
    cases = make_test_cases(config, models[0]["config"])
    logger = TestLogger(args.output_dir, config, models, cases)
    print(f"Testing {len(models)} models, {len(cases)} episodes per model\nOutput: {logger.run_dir}", flush=True)
    try:
        for model in models:
            method, seed, snapshot = model["method"], model["seed"], model["config"]
            set_seed(seed)

            agent_class = AGENT_MAP[method]
            agent = agent_class(dict(snapshot[agent_class.algorithm], **Observation.dimensions(),
                                     device=config.device), training=False)
            agent.load_model(model["checkpoint"])
            env = TransportEnv(cases[0]["formation"],
                               parameters(env_settings, **dict(snapshot["env"], time_limit=cases[0]["time_limit"])),
                               "human" if config.render else None, path_type=cases[0]["path_type"],
                               reference=parameters(reference_settings, **snapshot["reference"]),
                               disturbance_config=parameters(disturbance_settings, **snapshot["disturbance"]),
                               seed=config.seed)
            try:
                safety = SafetyFilter(env, parameters(safety_settings, **snapshot["safety"])) if method == "Ours" else None

                for index, case in enumerate(cases, 1):
                    env.config.time_limit = case["time_limit"]
                    metrics = run_episode(agent, env, safety, case)
                    row = logger.log_episode(model, case, metrics)
                    print(f"{method} seed{seed} [{index}/{len(cases)}] {case['case_id']} "
                          f"end={row['termination_reason']} return={row['env_return']:.3f} "
                          f"success={int(row['success'])} violation={int(row['coordination_violation'])} "
                          f"CE={row['attachment_mean']:.4f} TE={row['path_rmse']:.4f}", flush=True)
            finally:
                env.close()

        logger.save()
    finally:
        logger.close()
    print(f"Finished: {logger.run_dir / 'generalization_results.xlsx'}", flush=True)
    return logger.run_dir


if __name__ == "__main__":
    main()
