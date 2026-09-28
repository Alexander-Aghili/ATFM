"""Check the 20-line function limit in every tracked Python source file."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def tracked_sources(root=ROOT):
    result = subprocess.run(['git', 'ls-files', '-z', '--', '*.py'], cwd=root,
                            check=True, capture_output=True, text=True)
    return [root / name for name in result.stdout.split('\0') if name]


def violations(paths, limit=20):
    failures = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > limit:
                    failures.append(f'{path}:{node.lineno}: {node.name} spans {length} lines (limit {limit})')
    return failures


def main():
    failures = violations(tracked_sources())
    if failures:
        raise SystemExit('\n'.join(failures))
    print('All tracked Python functions are at most 20 physical lines.')


if __name__ == '__main__':
    main()
