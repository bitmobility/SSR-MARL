"""Robust barrier constraints with DOB compensation and error margins."""

import numpy as np


class RobustHOCBF:
    def __init__(self, model, error_bound):
        self.model = model

        self.error_bound = np.array(error_bound, dtype=float)  # m/s^2, rad/s

    def bounds(self, state, estimate):
        return self.model.bounds(state, estimate[:, 0], self.error_bound[0])

    def constraints(self, h, gradient, hessian, state, gains, estimate):
        k1, k2 = gains
        speed, heading = state["vehicle_speeds"], state["vehicle_headings"]
        n = len(speed)
        tangent = np.column_stack((np.cos(heading), np.sin(heading)))
        normal = np.column_stack((-np.sin(heading), np.cos(heading)))
        velocity, B = self.model.motion(state)
        blocks = gradient.reshape(-1, n, 2)

        coefficient = np.stack((np.einsum("mni,ni->mn", blocks, tangent),
                                np.einsum("mni,ni->mn", blocks, speed[:, None] * normal)), axis=2)
        compensation = np.einsum("mni,ni->m", coefficient, estimate)
        margin = np.sum(np.abs(coefficient) * self.error_bound, axis=(1, 2))
        h_dot = gradient @ velocity
        c = (np.einsum("i,mij,j->m", velocity, hessian, velocity)
             + (k1 + k2) * h_dot + k1 * k2 * h + compensation - margin)
        return gradient @ B, c, {
            "h": h, "h_dot": h_dot, "psi1": h_dot + k1 * h,
            "dob_compensation": compensation, "error_margin": margin}
