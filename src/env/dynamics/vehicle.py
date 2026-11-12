import numpy as np


class Vehicle:
    def __init__(self, position, heading=0.0, speed=0.0):
        self.position = np.array(position, dtype=float, copy=True)
        self.heading = float(heading)
        self.speed = float(speed)
        self.radius = 0.3
        self.lf = 0.25
        self.lr = 0.25
        self.max_speed = 4.0  # m/s
        self.max_acceleration = 1.0  # m/s^2
        self.max_steering = np.pi / 6
        self.action = np.zeros(2)

    def act(self, action):
        self.action = np.array(action, dtype=float, copy=True)

    def step(self, dt, response=(1.0, 1.0, 0.0)):
        acceleration, steering = self.action
        acceleration_gain, yaw_gain, drag = response
        acceleration *= acceleration_gain
        if drag == 0:
            speed = np.clip(self.speed + acceleration * dt, -self.max_speed, self.max_speed)
            accelerating_time = dt if acceleration == 0 else (speed - self.speed) / acceleration
            distance = ((self.speed + speed) * accelerating_time / 2
                        + speed * (dt - accelerating_time))
        else:
            equilibrium = acceleration / drag
            free_speed = self.speed + (equilibrium - self.speed) * -np.expm1(-drag * dt)
            speed = np.clip(free_speed, -self.max_speed, self.max_speed)
            accelerating_time = dt
            if speed != free_speed:
                accelerating_time = -np.log1p(-(speed - self.speed)
                                              / (equilibrium - self.speed)) / drag
            distance = (equilibrium * accelerating_time
                        + (self.speed - equilibrium) * -np.expm1(-drag * accelerating_time) / drag
                        + speed * (dt - accelerating_time))
        beta = np.arctan(self.lr / (self.lf + self.lr) * np.tan(steering))
        half_turn = yaw_gain * distance * np.sin(beta) / (2 * self.lr)
        direction = self.heading + half_turn
        travel = distance * np.sinc(half_turn / np.pi)
        residual_integral = np.array([
            speed - self.speed - self.action[0] * dt,
            (yaw_gain - 1) * distance * np.sin(beta) / self.lr,
        ])
        self.position += travel * np.array([np.cos(direction), np.sin(direction)])
        self.heading += 2 * half_turn
        self.speed = float(speed)
        return residual_integral
