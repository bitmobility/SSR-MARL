import numpy as np

from ..dynamics.payload import rotation


class Observation:
    local_dim = 9
    obs_dim = 21
    agent_state_dim = 13
    global_state_dim = 12
    action_dim = 2
    geometry_scale = 2.0
    speed_scale = 4.0
    yaw_rate_scale = 1.0
    fields = {
        "speed": slice(0, 1), "relative_heading": slice(1, 3),
        "anchor": slice(3, 5), "attachment_error": slice(5, 7),
        "previous_action": slice(7, 9), "payload_motion": slice(9, 12),
        "path_position": slice(12, 14), "path_heading": slice(14, 16),
        "preview_position": slice(16, 18), "preview_heading": slice(18, 20),
        "team_size": slice(20, 21),
    }

    def __init__(self, config, reference_config):
        self.config = config
        self.reference_config = reference_config

    @classmethod
    def dimensions(cls):
        return {key: getattr(cls, key) for key in
                ("obs_dim", "agent_state_dim", "global_state_dim", "action_dim")}

    def observe(self, vehicles, payload, tracking, *,
                previous_state=None, elapsed_time=0.0):
        positions = np.array([vehicle.position for vehicle in vehicles])
        headings = np.array([vehicle.heading for vehicle in vehicles])
        actions = np.array([vehicle.action for vehicle in vehicles])
        speeds = np.array([vehicle.speed for vehicle in vehicles])
        action_high = np.array([[v.max_acceleration, v.max_steering] for v in vehicles])
        center, heading = payload.position, payload.heading
        R = rotation(heading)
        anchors = payload.local_anchors
        relative_heading = headings - heading
        errors = (positions - center) @ R - anchors
        local = np.column_stack((speeds / self.speed_scale,
                                 np.sin(relative_heading), np.cos(relative_heading),
                                 anchors / self.geometry_scale, errors / self.config.attachment_limit,
                                 actions / action_high))


        directions = np.column_stack((np.cos(relative_heading), np.sin(relative_heading)))
        velocity = np.mean(speeds[:, None] * directions, axis=0)
        yaw_rate = 0.0
        if previous_state is not None and elapsed_time > 0:
            velocity = (center - previous_state["payload_position"]) @ R / elapsed_time
            angle = heading - previous_state["payload_heading"]
            yaw_rate = np.arctan2(np.sin(angle), np.cos(angle)) / elapsed_time

        projection = tracking["projection"]
        path_angle = projection.heading - heading
        preview_features = np.zeros(4)
        if tracking["preview_valid"]:
            preview = tracking["preview"]
            preview_angle = preview.heading - heading
            preview_features[:2] = ((preview.position - center) @ R
                                    / self.reference_config.lookahead_distance)
            preview_features[2:] = np.sin(preview_angle), np.cos(preview_angle)
        common = np.concatenate((
            velocity / self.speed_scale, [yaw_rate / self.yaw_rate_scale],
            (projection.position - center) @ R / self.config.path_error_limit,
            [np.sin(path_angle), np.cos(path_angle)], preview_features,
            [len(vehicles) / 4.0]))
        return np.concatenate((local, np.broadcast_to(common, (len(vehicles), len(common)))),
                              axis=1).astype(np.float32)

    def critic_state(self, obs, vehicles, payload, dob_state):
        prediction = dob_state["predicted_state"]
        estimate = dob_state["estimate"]
        predicted_angle = prediction[:, 1] - payload.heading
        acceleration_high = np.array([v.max_acceleration for v in vehicles])
        dob_features = np.column_stack((prediction[:, 0] / self.speed_scale,
                                        np.arctan2(np.sin(predicted_angle), np.cos(predicted_angle)) / np.pi,
                                        estimate[:, 0] / acceleration_high,
                                        estimate[:, 1] / self.yaw_rate_scale))
        return {"agents": np.concatenate((obs[:, :self.local_dim], dob_features), axis=1).astype(np.float32),
                "global": obs[0, self.local_dim:].copy()}
