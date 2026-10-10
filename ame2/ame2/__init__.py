# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME2 extension package."""

import importlib.util as _ilu

# Isaac Lab is required for the tasks/sensors. Without it (e.g. offline mapping-model
# training or unit tests) only the pure-torch sub-packages (``ame2.mapping``,
# ``ame2.sensors.neural_mapping_utils``) are importable.
if _ilu.find_spec("isaaclab") is not None:
    # Register Gym environments.
    from .tasks import *

    # Register UI extensions.
    from .ui_extension_example import *
