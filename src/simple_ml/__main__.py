import argparse
from simple_ml.global_utils import count_code_lines_from_file
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Count lines command
    count_lines_parser = subparsers.add_parser(
        "count", help="Count lines in a Python file"
    )
    count_lines_parser.add_argument("file", type=str, help="Path to the Python file")

    args = parser.parse_args()

    if args.command == "count":
        file_path = args.file
        lines = count_code_lines_from_file(Path(file_path))
        print(f"Number of non-empty, non-comment lines in {file_path}: {lines}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
