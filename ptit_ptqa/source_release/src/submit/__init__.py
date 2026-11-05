"""Submission CLI package."""

def main():
    from src.submit.cli import main as _main
    return _main()

def parse_args(args=None):
    from src.submit.cli import parse_args as _parse_args
    return _parse_args(args)

__all__ = ["main", "parse_args"]
