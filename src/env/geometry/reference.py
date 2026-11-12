from Code.config import parameters

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.integrate import cumulative_trapezoid

import Code.config.env_cfgs.reference as reference_settings


@dataclass(frozen=True)
class PathPose:
    s: float
    position: np.ndarray
    heading: float


@dataclass(frozen=True)
class PathProjection(PathPose):
    distance: float


def _geometry(points, first, second, third):
    speed = np.linalg.norm(first, axis=1)
    cross12 = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    cross13 = first[:, 0] * third[:, 1] - first[:, 1] * third[:, 0]
    dot12 = np.sum(first * second, axis=1)
    curvature = cross12 / speed**3
    curvature_rate = cross13 / speed**4 - 3 * cross12 * dot12 / speed**6
    return np.column_stack((points, np.arctan2(first[:, 1], first[:, 0]),
                            curvature, curvature_rate))


def _bezier(control, parameter):
    degree = len(control) - 1
    basis = np.column_stack([
        comb(degree, i) * parameter**i * (1 - parameter)**(degree - i)
        for i in range(degree + 1)
    ])
    return basis @ control


class ReferencePath:
    tolerance = 1e-9

    def __init__(self, path_type="line", config=None):
        self.path_type = path_type
        self.config = config if config is not None else parameters(reference_settings)
        cfg = self.config
        for name in ("sample_spacing", "lookahead_distance", "line_length",
                     "turn_entry_length", "turn_span", "turn_exit_length",
                     "sine_wavelength", "sine_periods", "arc_radius", "generalization_turn_radius",
                     "omega_entry_length", "z_straight_length", "z_diagonal_length"):
            if getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < cfg.steering_utilization <= 1:
            raise ValueError("steering_utilization must be in (0, 1]")
        if path_type == "line":
            geometry = self._line((0, 0), cfg.line_length, 0.0)
        elif path_type == "left_turn":
            geometry = self._left_turn()
        elif path_type == "sine":
            geometry = self._sine()
        elif path_type == "arc":
            geometry = self._arc()
        elif path_type == "omega":
            radius = cfg.generalization_turn_radius
            geometry = self._join_segments([
                self._line((0, 0), cfg.omega_entry_length, 0),
                self._smooth_turn(2 * np.pi / 3, radius),

                self._smooth_turn(-4 * np.pi / 3, 1.5 * radius),
                self._smooth_turn(2 * np.pi / 3, radius),
                self._line((0, 0), cfg.omega_entry_length, 0)])
        elif path_type == "z":
            radius = cfg.generalization_turn_radius
            geometry = self._join_segments([
                self._line((0, 0), cfg.z_straight_length, 0),
                self._smooth_turn(-3 * np.pi / 4, radius),
                self._line((0, 0), cfg.z_diagonal_length, 0),
                self._smooth_turn(3 * np.pi / 4, radius),
                self._line((0, 0), cfg.z_straight_length, 0)])
        else:
            raise ValueError(f"Unknown path type: {path_type}; choose line, left_turn, sine, arc, omega or z")

        self.points = geometry[:, :2] + np.asarray(cfg.origin)
        self._headings = np.unwrap(geometry[:, 2])
        self._curvature = geometry[:, 3]
        self._curvature_rate = geometry[:, 4]
        self._vectors = np.diff(self.points, axis=0)
        self._segment_lengths = np.linalg.norm(self._vectors, axis=1)
        self._s = np.r_[0.0, np.cumsum(self._segment_lengths)]
        self.length = float(self._s[-1])

    def _grid(self, end, max_derivative=1.0):
        count = int(np.ceil(end * max_derivative / self.config.sample_spacing))
        return np.linspace(0.0, end, count + 1)

    def _line(self, start, length, heading):
        distance = self._grid(length)
        direction = np.array([np.cos(heading), np.sin(heading)])
        points = np.asarray(start) + distance[:, None] * direction
        return np.column_stack((points, np.full(len(points), heading),
                                np.zeros((len(points), 2))))

    def _sine(self):
        cfg = self.config
        frequency = 2 * np.pi / cfg.sine_wavelength
        max_slope = abs(cfg.sine_amplitude * frequency)
        x = self._grid(cfg.sine_periods * cfg.sine_wavelength,
                       np.hypot(1.0, max_slope))
        phase = frequency * x
        y = cfg.sine_amplitude * np.sin(phase)
        dy = cfg.sine_amplitude * frequency * np.cos(phase)
        ddy = -frequency**2 * y
        dddy = -frequency**2 * dy
        return _geometry(np.column_stack((x, y)),
                         np.column_stack((np.ones_like(x), dy)),
                         np.column_stack((np.zeros_like(x), ddy)),
                         np.column_stack((np.zeros_like(x), dddy)))

    def _left_turn(self):
        cfg = self.config
        span = cfg.turn_span
        control = span * np.array([[0, 0], [0.25, 0], [0.5, 0],
                                   [1, 0.5], [1, 0.75], [1, 1]])
        first_control = 5 * np.diff(control, axis=0)
        second_control = 4 * np.diff(first_control, axis=0)
        third_control = 3 * np.diff(second_control, axis=0)
        parameter = self._grid(1.0, np.linalg.norm(first_control, axis=1).max())
        bend = _geometry(_bezier(control, parameter),
                         _bezier(first_control, parameter),
                         _bezier(second_control, parameter),
                         _bezier(third_control, parameter))
        bend[:, 0] += cfg.turn_entry_length
        entry = self._line((0, 0), cfg.turn_entry_length, 0.0)
        exit_line = self._line((cfg.turn_entry_length + span, span),
                               cfg.turn_exit_length, np.pi / 2)

        return np.concatenate((entry[:-1], bend, exit_line[1:]))

    def _arc(self):
        radius = self.config.arc_radius
        angle = self._grid(5 * np.pi / 3, radius)
        points = radius * np.column_stack((np.sin(angle), 1 - np.cos(angle)))
        return np.column_stack((points, angle, np.full(len(angle), 1 / radius), np.zeros(len(angle))))

    def _smooth_turn(self, angle, radius):
        peak = np.sign(angle) / radius
        length = abs(angle) * radius / 0.8
        distance = self._grid(length)
        ramp = 0.2 * length
        remaining = length - distance
        heading = np.where(
            distance < ramp,
            peak / 2 * (distance - ramp / np.pi * np.sin(np.pi * distance / ramp)),
            np.where(remaining < ramp,
                     angle - peak / 2 * (remaining - ramp / np.pi * np.sin(np.pi * remaining / ramp)),
                     peak * (distance - ramp / 2)))
        edge = np.minimum(distance, remaining)
        curvature = np.where(edge < ramp, peak / 2 * (1 - np.cos(np.pi * edge / ramp)), peak)
        rate = np.where(edge < ramp, peak * np.pi / (2 * ramp) * np.sin(np.pi * edge / ramp)
                        * np.where(distance < ramp, 1, -1), 0)
        tangent = np.column_stack((np.cos(heading), np.sin(heading)))
        points = cumulative_trapezoid(tangent, distance, axis=0, initial=0)
        return np.column_stack((points, heading, curvature, rate))

    @staticmethod
    def _join_segments(segments):
        endpoint, heading = np.zeros(2), 0.0
        pieces = []
        for segment in segments:
            c, s = np.cos(heading), np.sin(heading)
            segment[:, :2] = segment[:, :2] @ np.array([[c, s], [-s, c]]) + endpoint
            segment[:, 2] += heading
            pieces.append(segment if not pieces else segment[1:])
            endpoint, heading = segment[-1, :2], segment[-1, 2]
        return np.concatenate(pieces)

    def sample(self, s):
        s = float(np.clip(s, 0.0, self.length))
        index = min(np.searchsorted(self._s, s, side="right") - 1,
                    len(self._vectors) - 1)
        fraction = (s - self._s[index]) / self._segment_lengths[index]
        position = self.points[index] + fraction * self._vectors[index]
        heading = float(np.interp(s, self._s, self._headings))
        return PathPose(s, position, heading)

    def project(self, position):
        offsets = np.asarray(position) - self.points[:-1]
        fraction = np.clip(np.sum(offsets * self._vectors, axis=1)
                           / self._segment_lengths**2, 0.0, 1.0)
        residual = offsets - fraction[:, None] * self._vectors
        distance_squared = np.sum(residual**2, axis=1)
        index = int(np.argmin(distance_squared))
        pose = self.sample(self._s[index] + fraction[index] * self._segment_lengths[index])
        return PathProjection(pose.s, pose.position, pose.heading,
                              float(np.sqrt(distance_squared[index])))

    def preview(self, position, heading, projection=None):
        position = np.asarray(position)
        if projection is None:
            projection = self.project(position)
        forward = np.array([np.cos(heading), np.sin(heading)])
        require_heading = self.path_type not in self.config.path_only_preview_paths
        radius = self.config.lookahead_distance
        offset = self.points[:-1] - position
        # |offset + fraction * segment|^2 = radius^2.
        a = self._segment_lengths**2
        b = np.sum(offset * self._vectors, axis=1)
        c = np.sum(offset**2, axis=1) - radius**2
        discriminant = b**2 - a * c
        indices = np.flatnonzero(discriminant >= 0)
        roots = ((-b[indices, None] + np.sqrt(discriminant[indices, None]) * [-1, 1])
                 / a[indices, None])
        s = self._s[indices, None] + roots * self._segment_lengths[indices, None]
        valid = ((roots >= 0) & (roots <= 1)
                 & (s > projection.s + self.tolerance))
        if require_heading:
            relative = offset[indices, None, :] + roots[:, :, None] * self._vectors[indices, None, :]
            valid &= relative @ forward > self.tolerance
        if np.any(valid):
            return self.sample(np.min(s[valid]))

        endpoint = self.points[-1] - position
        if (self.length > projection.s + self.tolerance
                and np.linalg.norm(endpoint) <= radius + self.tolerance
                and (not require_heading or endpoint @ forward > self.tolerance)):
            return self.sample(self.length)
        return None

    def check_feasibility(self, anchors, vehicles, *, yaw_gain_min=1.0):
        anchors = np.asarray(anchors)
        curvature = self._curvature[:, None]
        forward = 1 - curvature * anchors[:, 1]
        if np.any(forward <= 0):
            vehicle_index = int(np.argwhere(forward <= 0)[0, 1])
            raise ValueError(f"{self.path_type}: vehicle_{vehicle_index} attachment trajectory folds inward")
        scale = np.hypot(forward, curvature * anchors[:, 0])
        anchor_curvature = ((curvature * scale**2
                             + self._curvature_rate[:, None] * anchors[:, 0]) / scale**3)
        max_curvature = np.max(np.abs(anchor_curvature), axis=0)

        lf = np.array([vehicle.lf for vehicle in vehicles])
        lr = np.array([vehicle.lr for vehicle in vehicles])
        steering_limit = self.config.steering_utilization * np.array(
            [vehicle.max_steering for vehicle in vehicles])
        beta_limit = np.arctan(lr / (lf + lr) * np.tan(steering_limit))
        curvature_limit = yaw_gain_min * np.sin(beta_limit) / lr
        if np.any(max_curvature > curvature_limit):
            vehicle_index = int(np.argmax(max_curvature / curvature_limit))
            raise ValueError(
                f"{self.path_type}: vehicle_{vehicle_index} required curvature "
                f"{max_curvature[vehicle_index]:.3f} exceeds the steering limit of "
                f"{curvature_limit[vehicle_index]:.3f} 1/m; increase the turn radius or path scale")

        beta = np.arcsin(lr * max_curvature / yaw_gain_min)
        return {"max_curvature": max_curvature,
                "max_steering": np.arctan((lf + lr) / lr * np.tan(beta)),
                "steering_limit": steering_limit}
