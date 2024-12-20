import importlib
import inspect
import os
import sys
from typing import Any

from torch import nn


def get_num_params(module: nn.Module) -> tuple[float, str]:
    num_params = sum([p.numel() for p in module.parameters() if p.requires_grad])
    metric = 'K'
    for metric in ['K', 'M', 'B']:
        num_params /= 1000
        if num_params < 1000:
            break
    return num_params, metric


def process_symbol(symbol: Any) -> None:
    if not inspect.isclass(symbol):
        return
    try:
        instance = symbol()
        if isinstance(instance, nn.Module):
            num_params, metric = get_num_params(instance)
            print(f'{instance.__class__.__name__}: {num_params:.2g}{metric}')
    except TypeError:
        pass


def main(path: str) -> None:
    if not path.endswith('.py'):
        raise ValueError(f'{path} not a python file.')

    root_path = os.path.abspath(os.path.curdir)
    sys.path.insert(0, root_path)
    module_name = path.replace(os.path.sep, '.')
    module_name = os.path.splitext(module_name)[0]
    module = importlib.import_module(module_name)
    for symbol in dir(module):
        if symbol.startswith('_'):
            continue
        symbol = getattr(module, symbol)
        process_symbol(symbol)


if __name__ == '__main__':
    main(sys.argv[1])

