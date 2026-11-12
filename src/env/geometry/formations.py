import numpy as np


def regular_polygon(n, radius=1.5):
    angles = np.arange(n) * 2 * np.pi / n
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def polygon_centroid(vertices):
    vertices = np.asarray(vertices, dtype=float)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    return ((vertices + following) * cross[:, None]).sum(axis=0) / (3 * cross.sum())


_triangle = regular_polygon(3)
_square = 1.5 / np.sqrt(2) * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
_rectangle = np.array([[-1.5, -.9], [1.5, -.9], [1.5, .9], [-1.5, .9]])
_pentagon, _hexagon = regular_polygon(5), regular_polygon(6)
_t = np.array([[-1.5, 1.5], [-1.5, .9], [-.3, .9], [-.3, -1.5],
               [.3, -1.5], [.3, .9], [1.5, .9], [1.5, 1.5]])
_l = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, -.9],
               [-.9, -.9], [-.9, 1.5], [-1.5, 1.5]])
_t_center, _l_center = polygon_centroid(_t), polygon_centroid(_l)


FORMATIONS = {
    "bar": (np.array([[-1.5, 0.0], [1.5, 0.0]]),
            np.array([[-1.5, -0.2], [1.5, -0.2], [1.5, 0.2], [-1.5, 0.2]])),
    "triangle": (_triangle, _triangle),
    "square": (_square, _square),
    "rectangle_4": (_rectangle, _rectangle),
    "rectangle_5": (np.vstack((_rectangle, [[0., 0.]])), _rectangle),
    "rectangle_6": (np.vstack((_rectangle, [[0., -.9], [0., .9]])), _rectangle),
    "pentagon_5": (_pentagon, _pentagon),
    "hexagon_6": (_hexagon, _hexagon),
    "t_3": (np.array([[-1.2, 1.2], [1.2, 1.2], [0., -1.2]]) - _t_center, _t - _t_center),
    "l_3": (np.array([[-1.2, -1.2], [1.2, -1.2], [-1.2, 1.2]]) - _l_center, _l - _l_center),
}
