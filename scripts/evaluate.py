"""Run a trained Ours, IPPO, MADDPG, or MAPPO-Lag policy on one configuration."""
import argparse
import csv
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    spec = ModuleSpec("Code", loader=None, is_package=True)
    spec.submodule_search_locations = [str(Path(__file__).resolve().parents[1])]
    sys.modules["Code"] = module_from_spec(spec)

import torch
import yaml

from Code.src.utils.evaluation import sampling_config
from Code.config import parameters
from Code.config.env_cfgs import transport, reference, disturbance, safety as safety_settings
from Code.src.alg.proposed import Proposed
from Code.src.alg.baselines.ippo import IPPO
from Code.src.alg.baselines.maddpg import MADDPG
from Code.src.alg.baselines.mappo_lag import MAPPOLag
from Code.src.cbf import SafetyFilter
from Code.src.env import FORMATIONS, Observation, TransportEnv
from Code.src.utils.metrics import TestEpisodeStats
from Code.src.utils.random import set_seed

AGENTS = {"Ours": Proposed, "IPPO": IPPO, "MADDPG": MADDPG, "MAPPO-Lag": MAPPOLag}


def run_episode(agent, env, safety=None, *, deterministic=True):
    obs, info = env.reset()
    if safety is not None:
        safety.reset(env)
    stats, hidden = TestEpisodeStats(), None
    while True:
        if agent.algorithm == "ippo":
            action, hidden = agent.choose_action(obs, hidden=hidden, deterministic=deterministic)
        elif agent.algorithm == "maddpg":
            action = agent.choose_action(obs)
        else:
            action = agent.choose_action(obs, deterministic=deterministic)
        state, nominal = env.get_state(), env.scale_action(action)
        executed, cbf = (safety.filter(state, nominal) if safety is not None else
                         (nominal, dict(correction_cost=0., fallback_used=False, status="disabled")))
        obs, reward, terminated, truncated, info = env.step(executed, policy_action=action)
        if safety is not None and info["valid_transition"]:
            safety.observer.update(state, info["executed_action"], info["full_state"], info["elapsed_time"])
        stats.update(reward, 0., info, cbf)
        env.render(phase=agent.algorithm, info=info)
        if env.render_closed:
            raise KeyboardInterrupt("Viewer closed")
        if terminated or truncated:
            return stats.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=AGENTS, default="Ours")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to config.yaml beside the checkpoints directory")
    parser.add_argument("--formation", choices=FORMATIONS, default="triangle")
    parser.add_argument("--path", choices=("line", "left_turn", "sine", "arc", "z", "omega"), default="line")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions (IPPO/MAPPO-Lag/Ours)")
    parser.add_argument("--render", action="store_true", help="Open the optional Pygame viewer")
    parser.add_argument("--output", type=Path, help="Optional per-episode CSV")
    args = parser.parse_args(argv)
    if args.episodes < 1 or (args.time_limit is not None and args.time_limit <= 0):
        parser.error("Episodes and time limit must be positive")
    if args.stochastic and args.method == "MADDPG":
        parser.error("MADDPG has a deterministic evaluation policy")
    snapshot_path = args.config or args.checkpoint.parent.parent / "config.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    set_seed(args.seed)
    cls = AGENTS[args.method]
    agent = cls(dict(snapshot[cls.algorithm], **Observation.dimensions(), device=args.device), training=False)
    agent.load_model(args.checkpoint)
    limit = args.time_limit or sampling_config(snapshot)["time_limits"].get(args.path, 180.)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        set_seed(seed)
        env = TransportEnv(args.formation, parameters(transport, **dict(snapshot["env"], time_limit=limit)),
                           "human" if args.render else None, path_type=args.path,
                           reference=parameters(reference, **snapshot["reference"]),
                           disturbance_config=parameters(disturbance, **snapshot["disturbance"]), seed=seed)
        try:
            safety = SafetyFilter(env, parameters(safety_settings, **snapshot["safety"])) if args.method == "Ours" else None
            result = run_episode(agent, env, safety, deterministic=not args.stochastic)
            row = dict(method=args.method, episode=episode + 1, seed=seed,
                       formation=args.formation, path=args.path, **result)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            env.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
