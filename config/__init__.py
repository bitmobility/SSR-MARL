from copy import deepcopy
from types import SimpleNamespace


def parameters(module, **values):
    settings = {name: value for name, value in vars(module).items() if not name.startswith("_")}
    return SimpleNamespace(**deepcopy(dict(settings, **values)))
