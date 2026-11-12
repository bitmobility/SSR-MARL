from .evaluation import sampling_config
from copy import deepcopy


def action_selection_labels(action_deterministic, methods):
    labels = {}
    for method in methods:
        deterministic = action_deterministic[method]
        if not isinstance(deterministic, bool):
            raise ValueError(f"{method}: ACTION_DETERMINISTIC must be True or False")
        if method == "MADDPG" and not deterministic:
            raise ValueError("MADDPG evaluation requires deterministic actions")
        labels[method] = "deterministic_policy" if deterministic else "stochastic_policy"
    return labels


def build_test_manifest(config, conditions, models, cases, action_deterministic):
    methods = list(dict.fromkeys(model["method"] for model in models))
    snapshot = models[0]["config"]
    environment = {key: snapshot[key] for key in ("env", "reference", "disturbance")}
    environment["sampling"] = {key: sampling_config(snapshot)[key] for key in (
        "time_limits", "initial_lateral_range", "initial_heading_range")}
    safety = {}
    if "Ours" in methods:
        safety["Ours"] = next(model["config"]["safety"] for model in models if model["method"] == "Ours")
    return deepcopy(dict(
        schema_version=2, complete=False,
        settings=vars(config), conditions=conditions,
        action_selection=action_selection_labels(action_deterministic, methods),

        environment=environment, safety=safety,
        planned_episodes=len(models) * len(cases),
        seed_counts={method: sum(model["method"] == method for model in models) for method in methods},
        models=[dict(method=model["method"], seed=model["seed"], checkpoint=str(model["checkpoint"]))
                for model in models],
    ))
