"""Dump teacher obs terms (env 0..3) at reset and after N teacher-JIT steps; works on 2.x (Kit) and 3.0."""
import argparse, os, sys
p = argparse.ArgumentParser()
p.add_argument("--stack", choices=("old", "is3"), required=True)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--jit", required=True)
p.add_argument("--eval_dir", required=True)
if sys.argv[sys.argv.index("--stack") + 1] == "old":
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    a, rest = p.parse_known_args(); a.headless = True
    app = AppLauncher(a).app
    env_cfg = None
    import contextlib; ctx = contextlib.nullcontext()
else:
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(p)
    import ame2.tasks  # noqa
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
    a, rest = setup_preset_cli(p, sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    env_cfg, _ = resolve_task_config("Ame2-G1-Gaze-Student", None)
    ctx = launch_simulation(env_cfg, a)
with ctx:
    import torch
    sys.path.insert(0, a.eval_dir)
    from _isaac_env import make_eval_env
    kw = {} if env_cfg is None else {"env_cfg": env_cfg}
    env, cfg, _ = make_eval_env("Ame2-G1-Gaze-Student", 4, "cuda:0", 0, "A", None, "rough", 0.0, **kw)
    u = env.unwrapped
    net = torch.jit.load(a.jit, map_location=u.device).eval()
    obs, _ = env.reset()
    om = u.observation_manager
    def dump(tag):
        names = om.active_terms["teacher_prop"]; dims = om.group_obs_term_dim["teacher_prop"]
        x = obs["teacher_prop"][0]; i = 0
        for n, d in zip(names, dims):
            k = int(torch.tensor(d).prod()); print(f"OBS {tag} {n}: " + " ".join(f"{v:.3f}" for v in x[i:i+k].tolist()), flush=True); i += k
        m = obs["teacher_mapping"][0]
        print(f"MAP {tag} mean/min/max per ch: " + " | ".join(f"{m[...,c].mean():.3f}/{m[...,c].min():.3f}/{m[...,c].max():.3f}" for c in range(m.shape[-1])), flush=True)
        r = u.scene["robot"]; pos = r.data.root_pos_w; pos = pos.torch if hasattr(pos, "torch") else pos
        print(f"ROOT {tag} z={[round(float(z),3) for z in pos[:,2]]}", flush=True)
    dump("reset")
    with torch.inference_mode():
        for t in range(a.steps):
            act = net(obs["teacher_prop"], obs["teacher_mapping"])[0]
            if t == 0: print("ACT0 " + " ".join(f"{v:.3f}" for v in act[0].tolist()), flush=True)
            obs, _, term, trunc, _ = env.step(act)
            if t in (0, 9) or t == a.steps - 1:
                dump(f"t{t}")
            if term.any(): print(f"TERM at {t}: {term.tolist()}", flush=True)
    env.close()
if sys.argv[sys.argv.index("--stack") + 1] == "old":
    app.close()
