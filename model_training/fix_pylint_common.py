#!/usr/bin/env python3
"""
Script to fix common pylint issues in Python files.
This script handles trailing whitespace and missing final newlines.
"""

import os
import sys
from pathlib import Path

def fix_file(file_path):
    """Fix common pylint issues in a single file."""
    print(f"Processing {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file_handle:
        lines = file_handle.readlines()

    # Remove trailing whitespace
    lines = [line.rstrip() + '\n' for line in lines]

    # Ensure final newline
    if lines and not lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip() + '\n'

    with open(file_path, 'w', encoding='utf-8') as file_handle:
        file_handle.writelines(lines)

def find_python_files(directory):
    """Find all Python files in the given directory and its subdirectories."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)

def main():
    """Main function to process all Python files."""
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path('.')

    for file_path in find_python_files(base_dir):
        fix_file(file_path)

if __name__ == "__main__":
    main()
