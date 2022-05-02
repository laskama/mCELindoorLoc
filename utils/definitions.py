from pathlib import Path
import os


def get_project_root():
    return str(Path(__file__).parent.parent)


def get_cache_path():
    path = get_project_root() + "/cached/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path
