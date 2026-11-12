"""Training episodes with reproducible task and initial-state sampling."""

from copy import deepcopy

import numpy as np

from Code.config import parameters
import Code.config.env_cfgs.transport as env_settings
import Code.config.env_cfgs.sampling as sampling_settings
from ...utils.random import make_rng
from ..geometry.formations import FORMATIONS
from .transport_env import TransportEnv


class TrainingEnv(TransportEnv):
    def __init__(self, config=None, render_mode=None, *, sampling=None,
                 reference=None, disturbance_config=None, seed=0, env_id=0):
        self.sampling = sampling if sampling is not None else parameters(sampling_settings)
        if not self.sampling.path_types or not self.sampling.formations:
            raise ValueError("Training formations and paths must be nonempty")
        self.env_id = env_id
        self.episode_index = 0
        self.episode_seed = None
        self.disturbance_scale = 1.0
        config = deepcopy(config) if config is not None else parameters(env_settings)
        config.time_limit = self.sampling.time_limits[self.sampling.path_types[0]]
        super().__init__(self.sampling.formations[0], config, render_mode,
                         path_type=self.sampling.path_types[0], reference=reference, seed=seed,
                         disturbance_config=disturbance_config)
        self.disturbance.sample(self.n, scale=self.disturbance_scale)

    def reset(self, options=None):
        options = {} if options is None else options
        derived_seed = int(np.random.SeedSequence(
            self.seed, spawn_key=(4, self.env_id, self.episode_index)).generate_state(1)[0])
        self.episode_seed = int(options.get("episode_seed", derived_seed))
        self.episode_index += 1
        path = options.get("path_type")
        if path is None:
            path = str(make_rng(self.episode_seed, 0).choice(self.sampling.path_types))
        formation = options.get("formation")
        if formation is None:
            formation = str(make_rng(self.episode_seed, 1).choice(self.sampling.formations))
        self.config.time_limit = self.sampling.time_limits[path]
        self.disturbance_scale = float(options.get("disturbance_scale", 1.0))
        if not self.config.disturbance_enabled:
            self.disturbance_scale = 0.0
        model_errors = options.get("disturbance_parameters")
        if model_errors is None:
            model_errors = self.disturbance.sample(
                len(FORMATIONS[formation][0]), rng=make_rng(self.episode_seed, 2),
                scale=self.disturbance_scale)
        pose_rng = make_rng(self.episode_seed, 3)
        lateral = options.get("initial_lateral_offset",
                              pose_rng.uniform(*self.sampling.initial_lateral_range))
        heading = options.get("initial_heading_offset",
                              pose_rng.uniform(*self.sampling.initial_heading_range))
        return super().reset(formation=formation, path_type=path,
                             disturbance_parameters=model_errors,
                             initial_lateral_offset=lateral, initial_heading_offset=heading)

    def episode_metadata(self):
        return dict(formation=self.formation, n_agents=self.n, path_type=self.path_type,
                    episode_seed=self.episode_seed, disturbance_scale=self.disturbance_scale,
                    disturbance_parameters=self.disturbance.parameters.copy(),
                    initial_lateral_offset=self.initial_lateral_offset,
                    initial_heading_offset=self.initial_heading_offset)
