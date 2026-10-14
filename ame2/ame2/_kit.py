"""Detect whether an Isaac Sim / Kit app is running in this process.

``isaaclab`` can be importable (e.g. in the Isaac conda env) while Kit is not
started; importing the task/sensor modules then fails on ``omni``/``pxr``.
The Isaac-dependent sub-packages are therefore only auto-imported once Kit is
up (``AppLauncher`` / ``SimulationApp`` loads ``omni.kit.app``). Scripts that
launch the app import ``ame2.tasks`` explicitly afterwards, so registration is
unaffected; offline tools (mapping-model training, tests) never touch Kit.
"""

import sys


def kit_running() -> bool:
    if "omni.kit.app" not in sys.modules:
        return False
    try:
        return sys.modules["omni.kit.app"].get_app() is not None
    except Exception:
        return False
