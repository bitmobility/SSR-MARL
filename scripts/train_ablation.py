"""Train safety-filter, DOB and active-coordination ablations."""

import argparse
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    spec = ModuleSpec("Code", loader=None, is_package=True)
    spec.submodule_search_locations = [str(Path(__file__).resolve().parents[1])]
    sys.modules["Code"] = module_from_spec(spec)


from Code.scripts import train_proposed
from Code.config import parameters
import Code.config.train_ablation as train_settings
import Code.config.env_cfgs.safety as safety_settings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(train_settings.variants), nargs="+",
                        default=train_settings.default_variants)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dirs = []
    for name in args.variant:
        train_cfg = parameters(train_settings)
        variant = train_cfg.variants[name]
        del train_cfg.variants, train_cfg.default_variants
        train_cfg.experiment_name = f"{train_cfg.experiment_name}/{name}"
        safety_cfg = parameters(safety_settings, **variant["safety"])
        ablation = SimpleNamespace(variant=name, use_safety_layer=variant["use_safety_layer"],
                                   critic_dob=True)
        print(f"Ablation {name}", flush=True)
        run_dirs.extend(train_proposed.run_training(
            train_cfg, safety_config=safety_cfg, ablation=ablation))
    return run_dirs


if __name__ == "__main__":
    main()
