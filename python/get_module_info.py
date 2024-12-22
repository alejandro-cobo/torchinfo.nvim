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
    parser.add_argument('--detailed', action='store_true', help='Show information about children modules too.')
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
    for metric in ('', 'K', 'M', 'G'):
        if num < 1000:
            return fmt % num + metric
        num /= 1000
    return fmt % (num * 1000) + metric


def get_num_params(module: nn.Module) -> int:
    num_params = sum([p.numel() for p in module.parameters() if p.requires_grad])
    return num_params


def get_flops(module: nn.Module, gpu: int) -> int:
    device = 'cpu' if gpu < 0 else gpu
    module = module.to(device)
    module.eval()
    input = torch.randn(module.input_shape).to(device)
    flop_counter = FlopCounterMode(mods=module, display=False, depth=None)
    with flop_counter:
        module(input)
    return flop_counter.get_total_flops()


def print_symbol_info(symbol: Any, gpu: int, detailed: bool, indent_level: int = 0) -> None:
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
            info += ', ' + num_to_str(flops) + ' FLOPs'
        print(' ' * indent_level + f'{symbol.__class__.__name__}: {info}')

        if detailed:
            for module in symbol.children():
                print_symbol_info(module, gpu, False, 4)


def main(argv) -> None:
    args = parse_args(argv)
    if not args.file_path.endswith('.py'):
        raise ValueError(f'{args.file_path} not a python file.')

    module = get_module_from_file(args.file_path)
    for symbol in dir(module):
        if symbol.startswith('_'):
            continue
        symbol = getattr(module, symbol)
        print_symbol_info(symbol, args.gpu, args.detailed)


if __name__ == '__main__':
    main(sys.argv[1:])

