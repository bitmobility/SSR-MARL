"""Balanced sampling across all training configurations and paths."""

from itertools import product

from ...utils.random import make_rng


class BalancedCaseSampler:
    def __init__(self, sampling, seed):
        self.combinations = list(product(sampling.formations, sampling.path_types))
        if not self.combinations:
            raise ValueError("Training formations and paths must be nonempty")
        self.rng = make_rng(seed, 7)
        self.pending = []

    def next_options(self):
        if not self.pending:
            self.pending = self.rng.permutation(len(self.combinations)).tolist()
        formation, path = self.combinations[self.pending.pop()]
        return dict(formation=formation, path_type=path, disturbance_scale=1.0)
