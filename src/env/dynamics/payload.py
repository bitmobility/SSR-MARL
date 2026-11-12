import numpy as np


from ..geometry.formations import FORMATIONS


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


class Payload:
    def __init__(self, formation, position, heading=0.0):
        self.local_anchors, self.local_vertices = (a.copy() for a in FORMATIONS[formation])
        self.position = np.array(position, dtype=float, copy=True)
        self.heading = float(heading)
        self.fit_status = "ok"

    def world_anchors(self):
        return self.position + self.local_anchors @ rotation(self.heading).T

    def world_vertices(self):
        return self.position + self.local_vertices @ rotation(self.heading).T

    def attachment_errors(self, vehicle_positions):
        return np.asarray(vehicle_positions) - self.world_anchors()

    def fit(self, vehicle_positions):
        positions = np.asarray(vehicle_positions)
        local_center = self.local_anchors.mean(axis=0)
        center = positions.mean(axis=0)
        local = self.local_anchors - local_center
        relative = positions - center
        c = np.sum(local * relative)
        s = np.sum(local[:, 0] * relative[:, 1] - local[:, 1] * relative[:, 0])
        if np.hypot(c, s) < 1e-10:
            self.fit_status = "degenerate"
            return self.fit_status
        change = np.arctan2(s, c) - self.heading
        self.heading += np.arctan2(np.sin(change), np.cos(change))
        self.position = center - rotation(self.heading) @ local_center
        self.fit_status = "ok"
        return self.fit_status
