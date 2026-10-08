"""Smoke test that instantiates a CALVIN PlayTableSimEnv via hydra."""

import sys
from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main() -> int:
    calvin_env_dir = (
        Path(__file__).resolve().parents[2] / "third_party" / "calvin" / "calvin_env"
    )
    conf_dir = calvin_env_dir / "conf"

    print(f"hydra: {hydra.__version__}")
    print(f"conf:  {conf_dir}")

    with initialize_config_dir(config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_data_collection",
            overrides=[
                "use_vr=false",
                "cameras=static_and_gripper",
                "env.show_gui=false",
                "env.use_egl=false",
                "scene=calvin_scene_D",
            ],
        )

    env = instantiate(cfg.env)
    print(f"env:   {type(env).__name__}")

    obs = env.reset()
    print(f"reset OK: keys={sorted(obs.keys())}")

    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(3):
        obs, reward, done, info = env.step(action)
    print(f"step OK: reward={reward}, done={done}")

    rgb_static = obs["rgb_obs"].get("rgb_static")
    if rgb_static is not None:
        print(f"rgb_static shape: {rgb_static.shape}, dtype: {rgb_static.dtype}")

    env.close()
    print("CALVIN smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
