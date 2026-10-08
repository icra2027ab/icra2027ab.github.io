"""
Dataset processing utilities.

Helpers for building and querying embedding indices over large demonstration
datasets.  Implementation to be added.
"""


def task_description_from_path(task_file: str) -> str:
    """Derive a human-readable task description from an HDF5 filename.

    E.g. ``pick_up_the_mug_demo.hdf5`` → ``"pick up the mug"``.
    """
    import pathlib
    stem = pathlib.Path(task_file).stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")
