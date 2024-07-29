from pathlib import Path
import os
import errno


def get_project_root() -> Path:
    """
    Get the root directory of the project.

    This function assumes that the script calling it is located within the project's directory structure.
    It returns the parent directory of the script's location, which is considered the project root.

    Returns:
        Path: The path to the project's root directory.
    """
    return Path(__file__).parent.parent


def verify_path_exists(path: Path) -> None:
    """
    Verify if a given directory path exists, and create it if it does not.

    This function attempts to create the directory specified by `path`.
    If the directory already exists, it does nothing. If the directory cannot be created due to
    a reason other than it already existing, it raises an OSError.

    Args:
        path (Path): The directory path to verify or create.

    Raises:
        OSError: If the directory cannot be created for reasons other than it already existing.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
