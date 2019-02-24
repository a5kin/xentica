#!/usr/bin/env python3
"""
Simple script to print concole-wide title.

Ex. usage:
  title.py "My cool title"

"""
import shutil
import sys


def print_title(text):
    """Print given text as title."""
    nc, _ = shutil.get_terminal_size((80, 24))
    print("\033[1;37;40m")
    print("-" * nc)
    print((" " + text + " ").center(nc, "#"))
    print("-" * nc + "\033[0;37;40m")


if __name__ == "__main__":
    print_title(sys.argv[1])
