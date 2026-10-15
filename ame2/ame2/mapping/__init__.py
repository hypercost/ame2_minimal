"""Pure-torch neural-mapping building blocks (no Isaac Lab imports).

Everything in this sub-package only depends on ``torch`` so it can be unit
tested, used by the offline mapping-model training script, and reused by the
Isaac Lab sensor (:mod:`ame2.sensors.livox_neural_map_sensor`).

* :mod:`.grid`       -- point cloud -> local max-height grid (``scatter_reduce_``)
* :mod:`.models`     -- per-frame mapping models (dummy + gated-residual U-Net) and loaders
* :mod:`.losses`     -- beta-NLL and total-variation batch re-weighting
* :mod:`.augment`    -- synthetic terrains and input augmentation for model training
* :mod:`.pointcloud_noise` -- lidar point dropout / artifact model
"""

from .grid import LocalGridSpec, points_to_local_grid
from .losses import beta_nll_loss, mapping_loss, total_variation, tv_batch_weights
from .models import (
    DummyMappingModel,
    GatedResidualUNet,
    load_mapping_model,
    save_mapping_checkpoint,
    export_mapping_jit,
)
from .pointcloud_noise import apply_pointcloud_noise

LIVOX_HOLE = -3.0
"""Hole / unobserved sentinel (kept in sync with ``neural_mapping_utils.LIVOX_HOLE``)."""
