from pathlib import Path
import os
import errno


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def verify_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
