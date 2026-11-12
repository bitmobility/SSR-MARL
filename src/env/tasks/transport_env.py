"""Multi-vehicle transport dynamics, observations, rewards and termination."""

from Code.config import parameters

from copy import deepcopy

import numpy as np

from .observation import Observation
from ..dynamics.disturbance import Disturbance
from ..dynamics.payload import Payload
from ..geometry.formations import FORMATIONS
from ..geometry.reference import ReferencePath
from ..dynamics.vehicle import Vehicle
import Code.config.env_cfgs.transport as env_settings
import Code.config.env_cfgs.reference as reference_settings
from ...utils.random import make_rng


def angle_difference(first, second):
    angle = first - second
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class TransportEnv:
    def __init__(self, formation="bar", config=None, render_mode=None, *,
                 path_type="line", reference=None, seed=0, disturbance_config=None):
        self.config = config if config is not None else parameters(env_settings)
        self.reference_config = reference if reference is not None else parameters(reference_settings)
        self.path_type = path_type
        self.dt = self.config.dt
        self.substep = self.config.substep
        self.seed = seed
        self.formation = formation
        self.n = len(FORMATIONS[formation][0])
        self.render_mode = render_mode
        self.viewer = None
        self.render_closed = False
        self.disturbance = Disturbance(self.config.disturbance_enabled, self.n, make_rng(seed, 1),
                                       disturbance_config)
        self.observation = Observation(self.config, self.reference_config)
        TransportEnv.reset(self)

    def reset(self, *, formation=None, path_type=None, disturbance_parameters=None,
              initial_lateral_offset=0.0, initial_heading_offset=0.0):
        if formation is not None:
            self.formation = formation
        if path_type is not None:
            self.path_type = path_type
        self.n = len(FORMATIONS[self.formation][0])
        if disturbance_parameters is not None:
            if np.shape(disturbance_parameters) != (self.n, 3):
                raise ValueError("Response parameters must have shape (N, 3)")
            self.disturbance.set_parameters(disturbance_parameters)
        elif len(self.disturbance.parameters) != self.n:
            self.disturbance.sample(self.n)
        self.reference = ReferencePath(self.path_type, self.reference_config)
        start = self.reference.sample(0)
        self.initial_lateral_offset = float(initial_lateral_offset)
        self.initial_heading_offset = float(initial_heading_offset)
        normal = np.array([-np.sin(start.heading), np.cos(start.heading)])
        position = start.position + self.initial_lateral_offset * normal
        heading = start.heading + self.initial_heading_offset
        self.payload = Payload(self.formation, position, heading)
        self.vehicles = [Vehicle(position, heading=heading, speed=self.config.initial_speed)
                         for position in self.payload.world_anchors()]
        self.reference.check_feasibility(
            self.payload.local_anchors, self.vehicles,
            yaw_gain_min=self.disturbance.min_yaw_gain)
        self.time = 0.0
        self.render_closed = False
        if self.viewer is not None:
            if self.viewer.closed:
                self.viewer = None
            else:
                self.viewer.reset()
        diagnostics = self._diagnostics(self.vehicles, self.payload)
        diagnostics["elapsed_time"] = 0.0
        info = self._info(self.get_state(), diagnostics)
        info["reward_terms"] = {}
        obs = self.observation.observe(self.vehicles, self.payload, diagnostics)
        return obs, info

    def step(self, actions, *, policy_action=None):
        previous_state = self.get_state()
        duration = max(0.0, min(self.dt, self.config.time_limit - self.time))
        self.time, diagnostics = self._simulate(
            self.vehicles, self.payload, self.time, actions, duration)
        obs = self.observation.observe(self.vehicles, self.payload, diagnostics,
                                       previous_state=previous_state,
                                       elapsed_time=diagnostics["elapsed_time"])
        terminated = self.is_terminated(diagnostics)
        truncated = self.is_truncated(diagnostics)
        info = self._info(self.get_state(), diagnostics, policy_action, truncated)
        reward = self.reward(previous_state, info)
        return obs, reward, terminated, truncated, info

    def _simulate(self, vehicles, payload, time, actions, duration):
        for vehicle, action in zip(vehicles, actions):
            vehicle.act(action)
        steps = int(np.ceil(duration / self.substep))
        dt = duration / steps if steps else 0.0
        elapsed = 0.0
        residual_integral = np.zeros((self.n, 2))
        diagnostics = self._diagnostics(vehicles, payload)
        for _ in range(steps):
            if self.is_terminated(diagnostics):
                break
            for i, (vehicle, response) in enumerate(zip(vehicles, self.disturbance.parameters)):
                residual_integral[i] += vehicle.step(dt, response)
            payload.fit([vehicle.position for vehicle in vehicles])
            elapsed += dt
            diagnostics = self._diagnostics(vehicles, payload)
        diagnostics["elapsed_time"] = elapsed
        diagnostics["disturbance"]["mean_total_residual"] = (
            residual_integral / elapsed if elapsed > 0 else residual_integral)
        return time + elapsed, diagnostics

    def _diagnostics(self, vehicles, payload):
        positions = np.array([vehicle.position for vehicle in vehicles])
        radii = np.array([vehicle.radius for vehicle in vehicles])
        first, second = np.triu_indices(len(vehicles), k=1)
        clearance = (np.linalg.norm(positions[first] - positions[second], axis=1)
                     - radii[first] - radii[second])
        projection = self.reference.project(payload.position)
        preview = self.reference.preview(payload.position, payload.heading, projection)
        errors = payload.attachment_errors(positions)
        norms = np.linalg.norm(errors, axis=1)
        diagnostics = {
            "vehicle_pair_clearance": clearance,
            "collision_pairs": [(f"vehicle_{i}", f"vehicle_{j}")
                                for i, j, distance in zip(first, second, clearance)
                                if distance <= 0],
            "projection": projection,
            "preview": preview,
            "preview_valid": preview is not None,
            "path_progress": projection.s,
            "path_error": projection.distance,
            "heading_error": angle_difference(payload.heading, projection.heading),
            "path_deviation": projection.distance > self.config.path_error_limit + 1e-9,
            "disturbance": self.disturbance.diagnostics(vehicles),
            "attachment_errors": errors,
            "attachment_norms": norms,
            "max_attachment_error": float(norms.max()),
            "mean_squared_attachment_error": float(np.mean(norms**2)),
            "coordination_h": self.config.attachment_limit**2 - norms**2,
            "coordination_violation": bool(np.any(norms > self.config.attachment_limit + 1e-9)),
            "payload_anchors": payload.world_anchors(),
            "payload_vertices": payload.world_vertices(),
            "fit_status": payload.fit_status,
            "numerical_failure": payload.fit_status == "degenerate",
            "valid_transition": payload.fit_status == "ok",
            "max_abs_speed": max(abs(vehicle.speed) for vehicle in vehicles),
        }
        diagnostics.update(self._termination(payload, diagnostics))
        return diagnostics

    def _termination(self, payload, diagnostics):
        events = []
        if diagnostics["numerical_failure"]:
            events.append("fit_degenerate")
        if diagnostics["collision_pairs"]:
            events.append("collision")
        if diagnostics["coordination_violation"]:
            events.append("coordination")
        if diagnostics["path_deviation"]:
            events.append("path_deviation")
        goal = self.reference.sample(self.reference.length)
        at_goal = (self.reference.length - diagnostics["path_progress"] <= self.config.goal_tolerance
                   and np.linalg.norm(payload.position - goal.position) <= self.config.goal_tolerance
                   and abs(angle_difference(payload.heading, goal.heading))
                   <= self.config.goal_heading_tolerance)
        if at_goal and not events:
            events.append("success")
        no_forward_reference = not diagnostics["preview_valid"] and "success" not in events
        if no_forward_reference:
            events.append("no_forward_reference")
        reason = events[0] if events else None
        return {"events": events, "termination_reason": reason, "success": reason == "success",
                "no_forward_reference": no_forward_reference,
                "tracking_failure": diagnostics["path_deviation"] or no_forward_reference}

    def reward(self, old_state, info):
        cfg = self.config
        state = info["full_state"]
        progress = info["path_progress"] - self.reference.project(old_state["payload_position"]).s
        action_change = np.mean(((state["action"] - old_state["action"]) / self.action_high)**2)
        coordination_error = np.mean((info["attachment_norms"] / cfg.attachment_limit)**2)
        elapsed = info["elapsed_time"]
        terminal_reward = 0.0
        if info["success"]:
            terminal_reward = cfg.success_reward
        elif self.is_terminated(info) and not info["numerical_failure"]:
            terminal_reward = cfg.failure_reward
        terms = {
            "progress": cfg.progress_weight * progress,
            "path": -cfg.path_weight * (info["path_error"] / cfg.path_error_limit)**2 * elapsed,
            "heading": -cfg.heading_weight * (1 - np.cos(info["heading_error"])) * elapsed,
            "coordination": -cfg.coordination_weight * coordination_error * elapsed,
            "time": -cfg.time_weight * elapsed,
            "action_change": -cfg.action_change_weight * action_change,
            "terminal": terminal_reward,
        }
        if info["numerical_failure"]:
            terms = dict.fromkeys(terms, 0.0)
        info["reward_terms"] = terms
        return float(sum(terms.values()))

    def is_terminated(self, info):
        return info["termination_reason"] not in (None, "time_limit")

    def is_truncated(self, info, *, time=None):
        time = self.time if time is None else time
        return not self.is_terminated(info) and time >= self.config.time_limit - 1e-10

    def _info(self, state, diagnostics, policy_action=None, truncated=False):
        info = {
            **diagnostics,
            "formation": self.formation,
            "seed": self.seed,
            "n_agents": self.n,
            "path_type": self.path_type,
            "full_state": state,
            "policy_action": None,
            "nominal_action": None,
            "executed_action": state["action"].copy(),
        }
        if truncated:
            info.update(termination_reason="time_limit", events=["time_limit"])
        if policy_action is not None:
            info["policy_action"] = np.array(policy_action, dtype=float, copy=True)
            info["nominal_action"] = self.scale_action(policy_action)
        return info

    def scale_action(self, policy_action):
        return np.asarray(policy_action, dtype=float) * self.action_high

    def render(self, phase="", info=None):
        if self.render_mode is None or self.render_closed:
            return None
        if self.viewer is None:
            from ..rendering.graphics import EnvViewer
            self.viewer = EnvViewer(self)
        frame = self.viewer.display(phase, info)
        self.render_closed = self.viewer.closed
        return frame

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def action_high(self):
        return np.array([[vehicle.max_acceleration, vehicle.max_steering]
                         for vehicle in self.vehicles])

    @property
    def action_low(self):
        return -self.action_high

    @staticmethod
    def _snapshot(vehicles, payload, time):
        return {
            "vehicle_positions": np.array([vehicle.position for vehicle in vehicles]),
            "vehicle_headings": np.array([vehicle.heading for vehicle in vehicles]),
            "vehicle_speeds": np.array([vehicle.speed for vehicle in vehicles]),
            "payload_position": payload.position.copy(),
            "payload_heading": float(payload.heading),
            "action": np.array([vehicle.action for vehicle in vehicles]),
            "time": time,
        }

    def get_state(self):
        return self._snapshot(self.vehicles, self.payload, self.time)

    def predict(self, state, actions, dt=None):
        vehicles = deepcopy(self.vehicles)
        for i, vehicle in enumerate(vehicles):
            vehicle.position = state["vehicle_positions"][i].copy()
            vehicle.heading = float(state["vehicle_headings"][i])
            vehicle.speed = float(state["vehicle_speeds"][i])
        payload = deepcopy(self.payload)
        payload.position = np.array(state["payload_position"], dtype=float, copy=True)
        payload.heading = float(state["payload_heading"])
        payload.fit_status = "ok"
        duration = max(0.0, min(self.dt if dt is None else dt,
                               self.config.time_limit - state["time"]))
        time, diagnostics = self._simulate(vehicles, payload, state["time"], actions, duration)
        next_state = self._snapshot(vehicles, payload, time)
        return next_state, self._info(
            next_state, diagnostics, truncated=self.is_truncated(diagnostics, time=time))
