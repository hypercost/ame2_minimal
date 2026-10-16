# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# --- G1 tasks ---

gym.register(
    id="Ame2-G1",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Ame2-G1-Play",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Ame2-G1-Gaze",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze",
    },
)

gym.register(
    id="Ame2-G1-Gaze-Play",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze",
    },
)

# G1 gaze student: LSIO actor over the Mid360 livox neural map, distilled from
# the frozen gaze teacher JIT at ``NN_models/g1_gaze_teacher_isaaclab.jit``.
gym.register(
    id="Ame2-G1-Gaze-Student",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_Gaze_Student",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze_Student",
    },
)

gym.register(
    id="Ame2-G1-Gaze-Student-Play",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_Gaze_Student_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze_Student",
    },
)

# Gaze student on the dummy neural map (top-down ground-truth "point cloud" +
# identity mapping model with constant variance): the full student pipeline
# without a trained mapping model.
gym.register(
    id="Ame2-G1-Gaze-Student-DummyMap",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_Gaze_Student_DummyMap",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze_Student_DummyMap",
    },
)

gym.register(
    id="Ame2-G1-Gaze-Student-DummyMap-Play",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_Gaze_Student_DummyMap_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze_Student_DummyMap",
    },
)

# Teacher env + Mid360 + top-down label scanner for mapping-model data collection
gym.register(
    id="Ame2-G1-Mapping-Collect",
    entry_point=f"{__name__}.env:Ame2ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame2_env_cfg:Ame2EnvCfg_G1_MappingCollect",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg_G1_Gaze",
    },
)
