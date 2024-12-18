from pathlib import Path


def count_code_lines(code_str: str) -> int:
    lines = code_str.split("\n")
    count = 0
    for line in lines:
        line = line.strip()
        # Skip blank lines and comment lines
        if line and not line.startswith("#"):
            count += 1
    return count


def count_code_lines_from_file(file_path: Path) -> int:
    """Count non-empty, non-comment lines in a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        Number of code lines
    """
    return count_code_lines(file_path.read_text())
