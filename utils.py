import sys
import os
import os.path as osp
from typing import Callable, Optional, Union
from functools import partial, wraps
from colorama import Fore, Style
from time import time
from datetime import datetime

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
