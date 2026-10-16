"""AME-2 environment class (IsaacLab 3.0 compatibility shim)."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class Ame2ManagerBasedRLEnv(ManagerBasedRLEnv):
    """:class:`ManagerBasedRLEnv` whose ``_reset_idx`` always receives an index tensor.

    IsaacLab 3.0 passes ``env_ids=slice(None)`` on a full ``env.reset()``; the AME-2 MDP terms,
    command, curriculum and sensors (written for 2.x) index with / take ``len()`` of ``env_ids``.
    """

    def _reset_idx(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        return super()._reset_idx(env_ids)
