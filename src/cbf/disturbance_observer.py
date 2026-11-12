"""Online acceleration and yaw-rate model error estimation."""

import numpy as np


class DisturbanceObserver:
    def __init__(self, model, state, gain):
        self.model = model
        self.gain = gain
        self.reset(state)

    def reset(self, state):
        self.predicted_state = self.model.state_vector(state)[:, 2:].copy()
        self.estimate = np.zeros_like(self.predicted_state)
        self.prediction_error = np.zeros_like(self.predicted_state)
        self.interval_residual = np.zeros_like(self.predicted_state)
        self.time = float(state["time"])

    def update(self, state, executed_actions, next_state, dt):
        if dt == 0:
            return self.get_state()
        previous = self.model.state_vector(state)
        current = self.model.state_vector(next_state)

        nominal_rate = (self.model.derivative(previous, executed_actions)
                        + self.model.derivative(current, executed_actions))[:, 2:] / 2
        previous, current = previous[:, 2:], current[:, 2:]
        self.interval_residual = (current - previous) / dt - nominal_rate


        decay = np.exp(-self.gain * dt)
        integral = -np.expm1(-self.gain * dt) / self.gain
        error = (decay * (self.predicted_state - previous)
                 + integral * (self.estimate - self.interval_residual))
        self.predicted_state = current + error
        self.prediction_error = error
        self.estimate = -self.gain / np.expm1(self.gain * dt) * error
        self.time = float(next_state["time"])
        return self.get_state()

    def get_state(self):
        return {"time": self.time, "estimate": self.estimate.copy(),
                "predicted_state": self.predicted_state.copy(),
                "prediction_error": self.prediction_error.copy(),
                "interval_residual": self.interval_residual.copy()}
