import argparse
import importlib
import inspect
import os
import sys
from types import ModuleType
from typing import Any

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to python file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id to use to compute FLOPs')
    args = parser.parse_args(argv)
    return args


def get_module_from_file(file_path: str) -> ModuleType:
    root_path = os.path.abspath(os.path.curdir)
    sys.path.insert(0, root_path)
    module_name = file_path.replace(os.path.sep, '.')
    module_name = os.path.splitext(module_name)[0]
    module = importlib.import_module(module_name)
    return module


def num_to_str(num: float, fmt: str = '%.2f') -> str:
    for metric in ['', 'K', 'M', 'B']:
        if num / 1000 < 1:
            return fmt % num + metric
        num /= 1000
    return fmt % num + 'B'


def get_num_params(module: nn.Module) -> int:
    num_params = sum([p.numel() for p in module.parameters() if p.requires_grad])
    return num_params


def get_flops(model, gpu: int) -> int:
    device = 'cpu' if gpu < 0 else gpu
    model = model.to(device)
    model.eval()
    input = torch.randn(model.input_shape).to(device)
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        model(input)
    return flop_counter.get_total_flops()


def print_symbol_info(symbol: Any, gpu: int) -> None:
    if inspect.isclass(symbol):
        try:
            symbol = symbol()
        except TypeError:
            return
    if isinstance(symbol, nn.Module):
        num_params = get_num_params(symbol)
        info = num_to_str(num_params) + ' params'
        if hasattr(symbol, 'input_shape'):
            flops = get_flops(symbol, gpu)
            info += ', ' + num_to_str(flops) + ' FLOPS'
        print(f'{symbol.__class__.__name__}: {info}')


def main(argv) -> None:
    args = parse_args(argv)
    if not args.file_path.endswith('.py'):
        raise ValueError(f'{args.file_path} not a python file.')

    module = get_module_from_file(args.file_path)
    for symbol in dir(module):
        if symbol.startswith('_'):
            continue
        symbol = getattr(module, symbol)
        print_symbol_info(symbol, args.gpu)


if __name__ == '__main__':
    main(sys.argv[1:])

