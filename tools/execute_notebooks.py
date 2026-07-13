import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path, timeout: int = 600) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"Executed {path}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python tools/execute_notebooks.py NOTEBOOK [NOTEBOOK ...]")
    for item in sys.argv[1:]:
        execute_notebook(Path(item))


if __name__ == "__main__":
    main()
