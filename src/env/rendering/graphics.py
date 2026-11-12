import numpy as np
import pygame


class EnvViewer:
    COLORS = [(230, 126, 34), (16, 156, 119), (158, 87, 190), (219, 75, 99),
              (42, 117, 185), (155, 122, 36)]

    def __init__(self, env):
        self.env = env
        self.human = env.render_mode == "human"
        self.width = 1200
        self.height = 680
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 30)
        self.surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.closed = False
        self.show_collision_bounds = False
        if self.human:
            pygame.display.init()
            pygame.display.set_caption("Multi-vehicle Cooperative Transport")
            self.screen = pygame.display.set_mode((self.width, self.height))
        self.reset()

    def reset(self):
        env = self.env
        extent = max(np.linalg.norm(env.payload.local_vertices, axis=1).max(),
                     max(np.linalg.norm(anchor) + vehicle.radius
                         for anchor, vehicle in zip(env.payload.local_anchors, env.vehicles)))
        padding = extent + env.config.path_error_limit + env.config.attachment_limit + 3.0
        low, high = env.reference.points.min(axis=0) - padding, env.reference.points.max(axis=0) + padding
        self.bounds = (low[0], high[0], low[1], high[1])
        xmin, xmax, ymin, ymax = self.bounds
        self.scale = min((self.width - 100) / (xmax - xmin),
                         (self.height - 210) / (ymax - ymin))
        self.origin = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        self.pixel_center = np.array([self.width / 2, 110 + (self.height - 210) / 2])
        self.trajectory = []
        self.vehicle_trajectories = [[] for _ in self.env.vehicles]
        self.last_time = None

    def pos2pix(self, position):
        offset = (np.asarray(position) - self.origin) * [1, -1]
        return tuple(np.rint(self.pixel_center + self.scale * offset).astype(int))

    def text(self, message, position, color=(40, 51, 67), title=False):
        font = self.title_font if title else self.font
        self.surface.blit(font.render(message, True, color), position)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                self.show_collision_bounds = not self.show_collision_bounds
        return not self.closed

    def draw_reference(self, info):
        xmin, xmax, ymin, ymax = self.bounds
        for x in np.arange(np.ceil(xmin / 5) * 5, xmax, 5):
            pygame.draw.line(self.surface, (235, 239, 244),
                             self.pos2pix((x, ymin)), self.pos2pix((x, ymax)))
        for y in np.arange(np.ceil(ymin / 5) * 5, ymax, 5):
            pygame.draw.line(self.surface, (235, 239, 244),
                             self.pos2pix((xmin, y)), self.pos2pix((xmax, y)))
        points = [self.pos2pix(p) for p in self.env.reference.points]
        pygame.draw.lines(self.surface, (176, 161, 208), False, points, 2)
        pygame.draw.circle(self.surface, (31, 157, 102), points[-1],
                           max(3, round(self.env.config.goal_tolerance * self.scale)), 2)
        self.text("GOAL", np.array(points[-1]) + [-20, 16])
        projection = self.pos2pix(info["projection"].position)
        pygame.draw.line(self.surface, (110, 120, 130),
                         self.pos2pix(self.env.payload.position), projection, 1)
        if info["preview_valid"]:
            preview = self.pos2pix(info["preview"].position)
            pygame.draw.circle(self.surface, (222, 72, 72), preview, 5, 2)

    def draw_vehicle(self, vehicle, color):
        front, rear = vehicle.lf + 0.06, vehicle.lr + 0.06
        width = 1.2 * vehicle.radius
        tire_length, tire_width = 0.12, 0.055
        supersampling = 3
        scale = self.scale * supersampling
        extent = max(front, rear, width / 2) + tire_length
        size = max(9, int(np.ceil(2 * extent * scale)))
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size / 2

        def rect(x, y, length, breadth):
            return pygame.Rect(round(center + (x - length / 2) * scale),
                               round(center - (y + breadth / 2) * scale),
                               max(1, round(length * scale)), max(1, round(breadth * scale)))

        body = rect((front - rear) / 2, 0, front + rear, width)
        border = (35, 47, 62)
        pygame.draw.rect(sprite, color, body, border_radius=supersampling)
        pygame.draw.rect(sprite, border, body, supersampling, border_radius=supersampling)


        roof_color = tuple(round(component * 0.65) for component in color)
        pygame.draw.rect(sprite, roof_color, rect(0, 0, (front + rear) * 0.40, width * 0.68))
        pygame.draw.rect(sprite, (195, 223, 240), rect(front * 0.48, 0, 0.07, width * 0.65))
        for side in (-1, 1):
            pygame.draw.rect(sprite, (255, 249, 205),
                             rect(front - 0.035, side * width * 0.30, 0.045, width * 0.22))
            pygame.draw.rect(sprite, (185, 42, 52),
                             rect(-rear + 0.025, side * width * 0.30, 0.03, width * 0.22))

        for axle, angle in ((-vehicle.lr, 0.0), (vehicle.lf, vehicle.action[1])):
            tire = pygame.Surface((max(1, round(tire_length * scale)),
                                   max(1, round(tire_width * scale))), pygame.SRCALPHA)
            tire.fill(border)
            tire = pygame.transform.rotate(tire, float(np.degrees(angle)))
            for side in (-1, 1):
                position = (round(center + axle * scale),
                            round(center - side * width / 2 * scale))
                sprite.blit(tire, tire.get_rect(center=position))

        sprite = pygame.transform.rotate(sprite, float(np.degrees(vehicle.heading)))
        sprite = pygame.transform.smoothscale(
            sprite, (max(1, round(sprite.get_width() / supersampling)),
                     max(1, round(sprite.get_height() / supersampling))))
        point = self.pos2pix(vehicle.position)
        self.surface.blit(sprite, sprite.get_rect(center=point))
        if self.show_collision_bounds:
            pygame.draw.circle(self.surface, (100, 116, 139), point,
                               max(1, round(vehicle.radius * self.scale)), 1)

    def draw_transport(self):
        env = self.env
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.surface, (47, 112, 210), False,
                              [self.pos2pix(p) for p in self.trajectory], 3)
            for i, trajectory in enumerate(self.vehicle_trajectories):
                pygame.draw.lines(self.surface, self.COLORS[i % len(self.COLORS)], False,
                                  [self.pos2pix(p) for p in trajectory], 1)
        vertices = [self.pos2pix(p) for p in env.payload.world_vertices()]
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (67, 128, 213, 75), vertices)
        self.surface.blit(overlay, (0, 0))
        pygame.draw.polygon(self.surface, (43, 99, 173), vertices, 2)
        pygame.draw.circle(self.surface, (43, 99, 173), self.pos2pix(env.payload.position), 4)
        for i, (vehicle, anchor) in enumerate(zip(env.vehicles, env.payload.world_anchors())):
            point = self.pos2pix(vehicle.position)
            color = self.COLORS[i % len(self.COLORS)]
            self.draw_vehicle(vehicle, color)
            error = np.linalg.norm(vehicle.position - anchor)
            connection_color = (216, 46, 46) if error > env.config.attachment_limit else (227, 159, 37)
            endpoint = self.pos2pix(anchor)
            pygame.draw.line(self.surface, connection_color, point, endpoint, 3)
            pygame.draw.circle(self.surface, connection_color, endpoint, 4, 1)
            self.text(f"V{i}", np.array(point) + [-10, -30], color)

    def display(self, phase="", info=None):
        if self.human and not self.handle_events():
            return None
        env = self.env
        if self.last_time != env.time:
            self.trajectory.append(env.payload.position.copy())
            for trajectory, vehicle in zip(self.vehicle_trajectories, env.vehicles):
                trajectory.append(vehicle.position.copy())
            self.last_time = env.time
        if info is None:
            info = env._diagnostics(env.vehicles, env.payload)
            if env.is_truncated(info):
                info["termination_reason"] = "time_limit"
        self.surface.fill((255, 255, 255))
        self.text(f"COOPERATIVE TRANSPORT  /  {env.formation.upper()}  /  {env.n} VEHICLES", (30, 18), title=True)
        self.text(f"Time: {env.time:.2f} s    Phase: {phase or '-'}    Status: {info['termination_reason'] or 'running'}",
                  (30, 52))
        self.text(f"Max speed: {info['max_abs_speed']:.2f} m/s", (self.width - 250, 52))
        self.text(f"Max connection error: {info['max_attachment_error']:.3e} m / {env.config.attachment_limit:.2f} m"
                  f"    Path error: {info['path_error']:.3f} m / {env.config.path_error_limit:.1f} m", (30, 78))
        self.draw_reference(info)
        self.draw_transport()
        self.text("Purple: planned path    Blue: payload trace    Thin colors: vehicle traces", (30, self.height - 64))
        self.text("Gold: attachment error    Red ring: preview    Green ring: goal"
                  f"    H: collision circles ({'on' if self.show_collision_bounds else 'off'})    Esc: close",
                  (30, self.height - 36))
        if self.human:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(round(1 / env.dt))
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def close(self):
        if self.human:
            pygame.display.quit()
        self.closed = True
