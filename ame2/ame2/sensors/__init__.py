# IsaacLab 3.0: the sensors only need ``isaaclab`` (kitless OVPhysX works), not a running Kit app.
# Pure-torch helpers (``neural_mapping_utils``) stay importable without Isaac Lab installed.
try:
    from .mid360_lidar import (
        MID360_MOUNT_OFFSET_ROT,
        Mid360PatternCfg,
        Mid360RayCaster,
        Mid360RayCasterCfg,
        add_noise,
    )
    from .livox_neural_map_sensor import LivoxNeuralMapSensor, LivoxNeuralMapSensorCfg
except ImportError:  # no Isaac Lab in this interpreter
    pass
