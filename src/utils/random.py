import numpy as np


def make_rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


def set_seed(seed):
    import torch

    numpy_seed = int(np.random.SeedSequence(seed, spawn_key=(2,)).generate_state(1)[0])
    torch_seed = int(np.random.SeedSequence(seed, spawn_key=(3,)).generate_state(1)[0])
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
