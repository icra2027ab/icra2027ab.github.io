"""
Adapted from: https://github.com/phospho-app/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py


Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    Run from the repo root inside the LIBERO-PRO Python 3.8 venv
    (see docs/libero_pro_setup.md §3 and §7):

    source third_party/LIBERO-PRO/.venv/bin/activate
    export PYTHONPATH="$PWD/third_party/LIBERO-PRO:$PWD/src"

    python -m policy_abstraction.data.regenerator \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python -m policy_abstraction.data.regenerator \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir datasets/libero_spatial \
            --libero_target_dir datasets/libero_spatial_no_noops

"""

import argparse
import json
import os
import time
from typing import Any

import h5py
import numpy as np
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_real_depth_map
import robosuite.utils.transform_utils as T
import tqdm

import sys
# ---------------------------------------------------------------------------
# Make sure the policy_abstraction package is importable when running from
# the repo root or the openpi subdirectory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Setup LIBERO-PRO path. Defaults to <repo-root>/third_party/LIBERO-PRO; env
# var override is provided for non-standard checkouts. ``_REPO_ROOT`` above is
# misnamed (it's the package root, src/policy_abstraction/), so we compute the
# real repo root locally rather than reuse it.
_TRUE_REPO_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))
_libero_pro_path = os.getenv("LIBERO_PRO_PATH") or os.path.join(
    _TRUE_REPO_ROOT, "third_party", "LIBERO-PRO"
)
if not os.path.isdir(_libero_pro_path):
    raise ImportError(
        f"LIBERO-PRO not found at {_libero_pro_path}. "
        "Clone it under third_party/ (see docs/install_libero_pro.md) "
        "or set LIBERO_PRO_PATH to an existing checkout."
    )
sys.path.insert(0, _libero_pro_path)

from libero.libero import benchmark

sys.path.insert(0, os.path.join(_REPO_ROOT, "envs", "libero"))
from config import LIBERO_DUMMY_ACTION
from utils import get_libero_env



IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        if not args.yes:
            user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
            if user_input != 'y':
                exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = args.libero_target_dir + f"{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, IMAGE_RESOLUTION, seed=0)

        # Compute near/far depth-conversion parameters from the MuJoCo model
        # (constant per scene, needed to convert normalised depth → metric depth).
        env.reset()
        _depth_near = env.sim.model.vis.map.znear * env.sim.model.stat.extent
        _depth_far  = env.sim.model.vis.map.zfar  * env.sim.model.stat.extent

        # Build mapping from mask ID (i+1) → instance name for offline analysis.
        # env.segmentation_id_mapping is {i: name}; the raw mask value is i+1.
        _seg_name_map = {
            str(i + 1): name for i, name in env.segmentation_id_mapping.items()
        }

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data: Any = orig_data_file["data"]  # type: ignore[index]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")
        # Store depth params so offline analysis tools can reconstruct metric depth.
        grp.attrs["depth_near"] = _depth_near
        grp.attrs["depth_far"]  = _depth_far
        # Store segmentation ID → name mapping for offline analysis.
        grp.attrs["segmentation_name_mapping"] = json.dumps(_seg_name_map)

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            for _ in range(9):
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            agentview_depth = []
            eye_in_hand_depth = []

            agentview_segmentation = []
            eye_in_hand_segmentation = []

            eye_in_hand_extrinsic = []
            eye_in_hand_intrinsic = []
            agentview_extrinsic = []
            agentview_intrinsic = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # get camera parameters
                eye_in_hand_extrinsic.append(get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand"))
                eye_in_hand_intrinsic.append(get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", IMAGE_RESOLUTION, IMAGE_RESOLUTION))

                agentview_extrinsic.append(get_camera_extrinsic_matrix(env.sim, "agentview"))
                agentview_intrinsic.append(get_camera_intrinsic_matrix(env.sim, "agentview", IMAGE_RESOLUTION, IMAGE_RESOLUTION))



                #add segmentation
                agentview_segmentation.append(obs["agentview_segmentation_instance"])
                eye_in_hand_segmentation.append(obs["robot0_eye_in_hand_segmentation_instance"])

                # get depth images (convert from normalised MuJoCo zbuffer → metric metres)
                agentview_depth.append(get_real_depth_map(env.sim, obs["agentview_depth"]))
                eye_in_hand_depth.append(get_real_depth_map(env.sim, obs["robot0_eye_in_hand_depth"]))

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))

                obs_grp.create_dataset("agentview_depth", data=np.stack(agentview_depth, axis=0))
                obs_grp.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depth, axis=0))

                obs_grp.create_dataset("agentview_intrinsic", data=np.stack(agentview_intrinsic, axis=0))
                obs_grp.create_dataset("eye_in_hand_intrinsic", data=np.stack(eye_in_hand_intrinsic, axis=0))

                obs_grp.create_dataset("agentview_extrinsic", data=np.stack(agentview_extrinsic, axis=0))
                obs_grp.create_dataset("eye_in_hand_extrinsic", data=np.stack(eye_in_hand_extrinsic, axis=0))

                obs_grp.create_dataset("agentview_segmentation", data=np.stack(agentview_segmentation, axis=0))
                obs_grp.create_dataset("eye_in_hand_segmentation", data=np.stack(eye_in_hand_segmentation, axis=0))

                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")




def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):

    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Bypass confirmation prompt when target directory already exists")
    args = parser.parse_args()

    # Start data regeneration
    main(args)
