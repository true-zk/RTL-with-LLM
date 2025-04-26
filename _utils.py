import sys
import os
import os.path as osp
from typing import Callable, Optional, Union, List
from functools import partial, wraps
from colorama import Fore, Style
from time import time
from datetime import datetime

import torch
from torch import Tensor

from config import CMD_LOG_DIR


def print_color(text: str, color: str = "green") -> None:
    colors = {
        "green": Fore.GREEN,
        "red": Fore.RED,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
    }
    print(colors.get(color, Fore.WHITE) + text + Style.RESET_ALL)


print_danger = partial(print_color, color="red")
print_success = partial(print_color, color="green")
print_warning = partial(print_color, color="yellow")


def wrapper_log_str_len(fn: Callable):
    r"""Decorator to log the length of strings returned by a generator function.
    Returns a generator that yields strings and stores their lengths in a list.

    Example:

        @warpper_log_str_len
        def my_generator():
            yield "Hello"
            yield "World"

        gen = my_generator()
        for val in gen:
            print(val)
        print(gen.lens)  # Output: [5, 5]
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        lens = []
        def gen():
            for v in fn(*args, **kwargs):
                assert isinstance(v, str), f"Expected str, got {type(v)}"
                lens.append(len(v))
                yield v

        class GeneratorAttachLen:
            def __init__(self):
                self._gen = gen()
                self.lens = lens

            def __iter__(self):
                return self._gen

            def __next__(self):
                return next(self._gen)

        return GeneratorAttachLen()

    return wrapper


def wrapper_timer(func):
    """Decorator to measure the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print_success(f">>>>>Execution time: {end_time - start_time:.4f} seconds<<<<<")
        return result

    return wrapper


def wrapper_cmd_logger(func_or_path: Optional[Union[str, Callable]] = None):
    """Decorator to log the output of a function to a file."""

    default_path = osp.join(CMD_LOG_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")

    def decorator(func:Callable, log_path: str = None):
        @wraps(func)
        def wrapper(*args, **kwargs):
            path = log_path or default_path
            if not os.path.exists(osp.dirname(path)):
                os.makedirs(osp.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                old_stdout = sys.stdout
                sys.stdout = f  # redirect stdout to the file
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = old_stdout
            return result
        return wrapper

    if callable(func_or_path):
        # If the first argument is a callable, treat it as the function to decorate
        return decorator(func_or_path)
    else:
        # If the first argument is a string, treat it as the log path
        log_path = func_or_path
        return partial(decorator, log_path=log_path)


y2label_map_dict = {
    'tacm12k': {
        0: "KDD",
        1: "CIKM",
        2: "WWW",
        3: "SIGIR",
        4: "STOC",
        5: "MobiCOMM",
        6: "SIGMOD",
        7: "SIGCOMM",
        8: "SPAA",
        9: "ICML",
        10: "VLDB",
        11: "SOSP",
        12: "SODA",
        13: "COLT",
        14: "[UNK]",
    },
    'tlf2k': {
        0: 'country',
        1: 'electronic',
        2: 'hip-hop',
        3: 'jazz',
        4: 'latin',
        5: 'pop',
        6: 'punk',
        7: 'reggae',
        8: 'rock',
        9: 'metal',
        10: 'soul'
    }
}

label2y_map_dict = {
    'tacm12k': {
        "KDD": 0,
        "CIKM": 1,
        "WWW": 2,
        "SIGIR": 3,
        "STOC": 4,
        "MobiCOMM": 5,
        "SIGMOD": 6,
        "SIGCOMM": 7,
        "SPAA": 8,
        "ICML": 9,
        "VLDB": 10,
        "SOSP": 11,
        "SODA": 12,
        "COLT": 13,
        "[UNK]": 14,
    },
    'tlf2k': {
        'country': 0,
        'electronic': 1,
        'hip-hop': 2,
        'jazz': 3,
        'latin': 4,
        'pop': 5,
        'punk': 6,
        'reggae': 7,
        'rock': 8,
        'metal': 9,
        'soul': 10
    }
}


def llm_preds_2_enhence_vec(
    llm_preds_l: List[List[str]],
    dataset_name: str,
    repeat_l: List[int] = [5, 3, 2, 1, 1]
) -> Tensor:
    vec = []
    mapping = label2y_map_dict[dataset_name]

    for el in llm_preds_l:
        assert len(el) == len(repeat_l)
        tmp = [x for i, v in enumerate(el)
               for x in ([mapping[v]] * repeat_l[i] if v in mapping else [mapping['[UNK]']] * repeat_l[i])]
        vec.append(torch.tensor(tmp))

    return torch.stack(vec, dim=0)
