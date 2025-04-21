import os
from pathlib import Path
import tomli as tomllib

# Dynamically determine the project root, assuming config.py is in /general
ROOT_DIR = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT_DIR / "pyproject.toml"
PYPROJECT_DEFAULT = ROOT_DIR / "pyproject.default.toml"

class Config:
    config = None
    if not PYPROJECT.exists() and PYPROJECT_DEFAULT.exists():
        import shutil
        print(f"copying {PYPROJECT_DEFAULT} to {PYPROJECT} for first run")
        shutil.copyfile(PYPROJECT_DEFAULT, PYPROJECT)

    try:
        with open(PYPROJECT, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{PYPROJECT} not found")

    @classmethod
    def get(cls, *path):
        result = cls.config
        for item in path:
            result = result.get(item, None)
            if result is None:
                break
        return result
