"""Nominal vehicle model and transformed input bounds."""

import numpy as np
from scipy.linalg import block_diag


class NominalModel:
    def __init__(self, env, speed_margin):
        self.dt = env.dt
        self.lf = np.array([v.lf for v in env.vehicles])
        self.lr = np.array([v.lr for v in env.vehicles])
        self.max_speed = np.array([v.max_speed for v in env.vehicles])
        self.speed_limit = self.max_speed - speed_margin
        self.max_acceleration = np.array([v.max_acceleration for v in env.vehicles])
        steering = np.array([v.max_steering for v in env.vehicles])
        self.max_s = np.sin(np.arctan(self.lr / (self.lf + self.lr) * np.tan(steering)))
        self.input_scale = np.column_stack((self.max_acceleration, self.max_s)).ravel()

    def to_input(self, actions):
        actions = np.asarray(actions, dtype=float)
        beta = np.arctan(self.lr / (self.lf + self.lr) * np.tan(actions[:, 1]))
        return np.column_stack((actions[:, 0], np.sin(beta))).ravel()

    def to_action(self, inputs):
        inputs = np.asarray(inputs).reshape(-1, 2)
        steering = np.arctan((self.lf + self.lr) / self.lr
                             * np.tan(np.arcsin(inputs[:, 1])))
        return np.column_stack((inputs[:, 0], steering))

    @staticmethod
    def state_vector(state):
        return np.column_stack((state["vehicle_positions"], state["vehicle_speeds"],
                                state["vehicle_headings"]))

    def derivative(self, vehicle_states, actions):
        speed, heading = vehicle_states[:, 2], vehicle_states[:, 3]
        inputs = self.to_input(actions).reshape(-1, 2)
        return np.column_stack((speed * np.cos(heading), speed * np.sin(heading),
                                inputs[:, 0], speed / self.lr * inputs[:, 1]))

    def motion(self, state):
        heading = state["vehicle_headings"]
        speed = state["vehicle_speeds"]
        tangent = np.column_stack((np.cos(heading), np.sin(heading)))
        normal = np.column_stack((-np.sin(heading), np.cos(heading)))
        velocity = (speed[:, None] * tangent).ravel()
        blocks = np.stack((tangent, (speed ** 2 / self.lr)[:, None] * normal), axis=2)
        return velocity, block_diag(*blocks)

    def bounds(self, state, estimate=0.0, error=0.0):
        speed = state["vehicle_speeds"]
        lower_a = np.maximum(-self.max_acceleration,
                             (-self.speed_limit - speed) / self.dt - estimate + error)
        upper_a = np.minimum(self.max_acceleration,
                             (self.speed_limit - speed) / self.dt - estimate - error)
        return (np.column_stack((lower_a, -self.max_s)).ravel(),
                np.column_stack((upper_a, self.max_s)).ravel())
