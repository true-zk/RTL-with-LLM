from functools import partial
from colorama import Fore, Style


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
