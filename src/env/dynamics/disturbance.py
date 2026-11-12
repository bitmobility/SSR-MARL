from Code.config import parameters

import numpy as np

import Code.config.env_cfgs.disturbance as disturbance_settings


class Disturbance:
    nominal = np.array([1.0, 1.0, 0.0])

    def __init__(self, enabled, n_agents, rng, config=None):
        self.enabled = enabled
        self.config = config if config is not None else parameters(disturbance_settings)
        self.rng = rng
        self.sample(n_agents)

    def sample(self, n_agents, *, rng=None, scale=1.0):
        rng = self.rng if rng is None else rng
        cfg = self.config
        lower, upper = np.array([cfg.acceleration_gain_range,
                                 cfg.yaw_gain_range, cfg.drag_range]).T
        sampled = rng.uniform(lower, upper, size=(n_agents, 3))
        self.parameters = self.nominal + (scale if self.enabled else 0.0) * (sampled - self.nominal)
        return self.parameters.copy()

    def set_parameters(self, parameters):
        self.parameters = np.array(parameters, dtype=float, copy=True)

    @property
    def min_yaw_gain(self):
        return self.config.yaw_gain_range[0] if self.enabled else 1.0

    def diagnostics(self, vehicles):
        speed = np.array([v.speed for v in vehicles])
        heading = np.array([v.heading for v in vehicles])
        lf = np.array([v.lf for v in vehicles])
        lr = np.array([v.lr for v in vehicles])
        max_speed = np.array([v.max_speed for v in vehicles])
        actions = np.array([v.action for v in vehicles])
        beta = np.arctan(lr / (lf + lr) * np.tan(actions[:, 1]))
        yaw_rate = speed / lr * np.sin(beta)
        nominal = np.column_stack((speed * np.cos(heading), speed * np.sin(heading),
                                   actions[:, 0], yaw_rate))
        injected = np.column_stack(((self.parameters[:, 0] - 1) * actions[:, 0]
                                    - self.parameters[:, 2] * speed,
                                    (self.parameters[:, 1] - 1) * yaw_rate))
        acceleration = actions[:, 0] + injected[:, 0]
        outward = ((speed >= max_speed) & (acceleration > 0)
                   | (speed <= -max_speed) & (acceleration < 0))
        saturation = np.zeros_like(injected)
        saturation[:, 0] = np.where(outward, -acceleration, 0.0)
        total = injected + saturation
        actual = nominal.copy()
        actual[:, 2:] += total
        return {"enabled": self.enabled,
                "parameters": self.parameters.copy(),
                "nominal_derivative": nominal, "actual_derivative": actual,
                "injected_residual": injected,
                "saturation_residual": saturation, "total_residual": total}
