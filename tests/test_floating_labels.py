"""Regression: floating_boxes labels must not contradict the lidar (overhangs above the sensor are see-through)."""
import pytest
pytest.importorskip("warp")
from ame2.mapping.grid import LocalGridSpec
from ame2.mapping.warp_sampler import SamplerCfg, WarpMappingSampler


def test_floating_box_labels_consistent_with_lidar():
    cfg = SamplerCfg(families=("floating_boxes",), family_weights=(1.0,), procedural_size=20.0)
    s = WarpMappingSampler(LocalGridSpec(), cfg, device="cpu", seed=1)
    x, y, ok = s.sample(8, clean=True, family="floating_boxes")
    e = (x - y)[(x > -2.9) & ok & (x <= 0)]
    # before the fix: signed -1.0 m, 66% of observed cells > 0.3 m below the label
    assert e.mean().abs() < 0.15
    assert (e < -0.3).float().mean() < 0.1
