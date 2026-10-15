# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME2 extension package."""

from ._kit import kit_running as _kit_running

# Tasks/sensors need a running Kit app (Isaac Sim). Without it -- no Isaac Lab, or Isaac Lab
# installed but Kit not started (offline mapping-model training, unit tests) -- only the
# pure-torch sub-packages (``ame2.mapping``, ``ame2.sensors.neural_mapping_utils``) load.
# Launch scripts import ``ame2.tasks`` explicitly after AppLauncher, as before.
if _kit_running():
    # Register Gym environments.
    from .tasks import *

    # Register UI extensions.
    from .ui_extension_example import *
