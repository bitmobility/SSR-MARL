"""Coordination derivatives, high-order barriers and active coordination."""

import numpy as np


class CoordinationCBF:
    def __init__(self, env, gains):
        anchors = env.payload.local_anchors
        self.anchors = anchors - anchors.mean(axis=0)
        self.limit = env.config.attachment_limit
        self.k1, self.k2 = gains

    def constraints(self, state, velocity, B, robust=None, estimate=None):
        positions = state["vehicle_positions"]
        n = len(positions)
        relative = positions - positions.mean(axis=0)
        cosine = np.sum(self.anchors * relative)
        sine = np.sum(self.anchors[:, 0] * relative[:, 1]
                      - self.anchors[:, 1] * relative[:, 0])
        fit_scale = np.hypot(cosine, sine)
        if fit_scale < 1e-10:
            raise ValueError("Degenerate payload fit: cannot build the coordination HOCBF")
        rotation = np.array([[cosine, -sine], [sine, cosine]]) / fit_scale
        anchors = self.anchors @ rotation.T
        perpendicular = np.column_stack((-anchors[:, 1], anchors[:, 0]))
        errors = relative - anchors


        centering = np.eye(n) - np.ones((n, n)) / n
        J = np.kron(centering, np.eye(2))
        J -= np.outer(perpendicular.ravel(), perpendicular.ravel()) / fit_scale
        error_velocity = (J @ velocity).reshape(n, 2)
        omega = perpendicular.ravel() @ velocity / fit_scale
        scale_rate = anchors.ravel() @ velocity
        drift = omega ** 2 * anchors + 2 * omega * scale_rate / fit_scale * perpendicular
        M = (J @ B).reshape(n, 2, 2 * n)

        h = self.limit ** 2 - np.sum(errors ** 2, axis=1)
        if robust is not None:
            blocks = J.reshape(n, 2, 2 * n)
            gradient = -2 * np.einsum("ni,nij->nj", errors, blocks)
            angle_gradient = perpendicular.ravel() / fit_scale
            angle_hessian = -(np.outer(anchors.ravel(), perpendicular.ravel())
                              + np.outer(perpendicular.ravel(), anchors.ravel())) / fit_scale ** 2
            error_hessian = (anchors[:, :, None, None] * np.outer(angle_gradient, angle_gradient)
                             - perpendicular[:, :, None, None] * angle_hessian)
            hessian = -2 * (np.einsum("nki,nkj->nij", blocks, blocks)
                            + np.einsum("nk,nkij->nij", errors, error_hessian))
            A, c, diagnostics = robust.constraints(
                h, gradient, hessian, state, (self.k1, self.k2), estimate)
            heading, speed = state["vehicle_headings"], state["vehicle_speeds"]
            tangent = np.column_stack((np.cos(heading), np.sin(heading)))
            normal = np.column_stack((-np.sin(heading), np.cos(heading)))
            residual = estimate[:, :1] * tangent + estimate[:, 1:] * speed[:, None] * normal
            drift = drift + (J @ residual.ravel()).reshape(n, 2)
        else:
            h_dot = -2 * np.sum(errors * error_velocity, axis=1)
            A = -2 * np.einsum("ni,nij->nj", errors, M)
            c = (-2 * np.sum(error_velocity ** 2 + errors * drift, axis=1)
                 + (self.k1 + self.k2) * h_dot + self.k1 * self.k2 * h)
            diagnostics = {"h": h, "h_dot": h_dot, "psi1": h_dot + self.k1 * h}
        # e_ddot_hat = acceleration_matrix @ U + acceleration_drift.ravel().
        diagnostics.update(attachment_errors=errors, error_velocity=error_velocity,
                           acceleration_matrix=M.reshape(2 * n, 2 * n), acceleration_drift=drift)
        return A, c, diagnostics
