"""Joint DOB-CBF quadratic program and physical action recovery."""

from Code.config import parameters

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import osqp
from scipy import sparse

import Code.config.env_cfgs.safety as safety_settings
from .coordination_cbf import CoordinationCBF
from .disturbance_observer import DisturbanceObserver
from .nominal_model import NominalModel
from .robust_hocbf import RobustHOCBF


class SafetyFilter:
    def __init__(self, env, config=None):
        self.config = config if config is not None else parameters(safety_settings)
        self.coordination_gains = (self.config.robust_coordination_gains if self.config.robust
                                   else self.config.nominal_coordination_gains)
        self.dob_gain = self.config.dob_gain
        self.coordination_weight = self.config.coordination_weight
        self.coordination_kp = self.config.coordination_kp
        self.coordination_kd = self.config.coordination_kd
        self.reset(env)

    def reset(self, env):
        self.model = NominalModel(env, self.config.speed_margin)
        self.robust = RobustHOCBF(self.model, self.config.error_bound) if self.config.robust else None
        self.observer = DisturbanceObserver(self.model, env.get_state(), self.dob_gain)
        self.coordination = CoordinationCBF(env, self.coordination_gains)

    def filter(self, state, nominal_actions, *, fallback=True):
        nominal_actions = np.asarray(nominal_actions, dtype=float)
        nominal = self.model.to_input(nominal_actions)
        velocity, B = self.model.motion(state)
        dob = self.observer.get_state()
        estimate = dob["estimate"]
        A, c, coordination = self.coordination.constraints(
            state, velocity, B, self.robust, estimate)
        if self.robust is None:
            lower, upper = self.model.bounds(state)
        else:
            lower, upper = self.robust.bounds(state, estimate)

        scale = self.model.input_scale
        acceleration_scale = np.repeat(self.model.max_acceleration, 2)
        D = coordination["acceleration_matrix"] * scale / acceleration_scale[:, None]
        b = (coordination["acceleration_drift"]
             + self.coordination_kd * coordination["error_velocity"]
             + self.coordination_kp * coordination["attachment_errors"]).ravel() / acceleration_scale
        P = np.eye(len(nominal)) + self.coordination_weight * D.T @ D
        q = -nominal / scale + self.coordination_weight * D.T @ b
        failure = None
        if np.any(lower > upper):
            status = "bounds_conflict"
            failure = "Conflicting action and speed bounds"
        else:
            row_scale = np.maximum(np.linalg.norm(A * scale, axis=1), 1e-3)
            matrix = sparse.vstack((sparse.csc_matrix(A * scale / row_scale[:, None]),
                                    sparse.eye(len(nominal), format="csc")), format="csc")
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(P), q=q,
                         A=matrix, l=np.concatenate((-c / row_scale, lower / scale)),
                         u=np.concatenate((np.full(len(c), np.inf), upper / scale)),
                         verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=20000,
                         rho=0.1, adaptive_rho=False, polishing=True)

            output = StringIO()
            try:
                with redirect_stdout(output):
                    result = solver.solve(raise_error=False)
            finally:
                for line in output.getvalue().splitlines(keepends=True):
                    if line.rstrip("\r\n") != "Polishing not needed - no active set detected at optimal point":
                        print(line, end="")
            status = result.info.status
            if status not in ("solved", "solved inaccurate"):
                failure = status
            else:
                safe = result.x * scale
                residual = min(np.min(A @ safe + c), np.min(safe - lower), np.min(upper - safe))
                if not np.all(np.isfinite(safe)) or residual < -1e-6:
                    status = "invalid_solution"
                    failure = f"Nonfinite action or excessive constraint residual: {residual:.3e}"
                else:
                    actions = self.model.to_action(safe)

        if failure is not None:
            if not fallback:
                raise RuntimeError(f"CBF QP failed: {failure}")
            safe = nominal.copy()
            actions = nominal_actions.copy()

        coordination["hocbf_margin"] = A @ safe + c
        coordination["tracking_residual"] = (D @ (safe / scale) + b).reshape(-1, 2)
        info = {"status": status, "robust": self.robust is not None,
                "fallback_used": failure is not None, "failure_reason": failure,
                "barrier_set_satisfied": bool(np.all(coordination["h"] >= 0)
                                               and np.all(coordination["psi1"] >= 0)),
                "input_lower": lower, "input_upper": upper,
                "nominal_action": nominal_actions.copy(),
                "safe_action": None if failure is not None else actions.copy(),
                "filtered_action": actions.copy(), "action_delta": actions - nominal_actions,
                "correction_cost": float(np.mean(((safe - nominal) / self.model.input_scale) ** 2)),
                "coordination_cost": float(np.mean(coordination["tracking_residual"] ** 2)),
                "coordination": coordination,
                "dob": dob}
        return actions, info
