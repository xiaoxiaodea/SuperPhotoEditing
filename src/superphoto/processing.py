"""Lightroom's active photo/mask state is shared across whole edit workflows."""
from functools import wraps
from threading import RLock

lightroom_workflow_lock = RLock()


def serialized_processing(function):
    @wraps(function)
    def run(*args, **kwargs):
        with lightroom_workflow_lock:
            return function(*args, **kwargs)
    return run
