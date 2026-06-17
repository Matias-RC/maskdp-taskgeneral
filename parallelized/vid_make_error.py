import glob
import os
os.environ["MUJOCO_GL"] = "egl"
import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Force 64-bit precision for consistency
jax.config.update("jax_enable_x64", True)

# 1. Setup Model & Native Renderer for video capture
xml_path = "custom_dmc_tasks/walker.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)

# Initialize renderer (Height and Width for the video frame)
width, height = 640, 480
renderer = mujoco.Renderer(mj_model, height=height, width=width)

# 2. Load Data
expert_data_path = "maskdp_data/maskdp_train/walker/expert/walker_walk/train"
npz_files = glob.glob(os.path.join(expert_data_path, "*.npz"))
demo = np.load(npz_files[0])

expert_physics = demo["physics"]
expert_actions = jnp.array(demo["action"][:-1, :], dtype=jnp.float64)

# 3. Reconstruct Dataset States (The Truth)
dataset_qpos = expert_physics[1:, :9]
dataset_qvel = expert_physics[1:, 9:]

# 4. Simulate MJX Rollout (The Divergent Path)
single_mjx_data = mjx.put_data(mj_model, mj_data).replace(
    qpos=jnp.array(expert_physics[0, :9]), qvel=jnp.array(expert_physics[0, 9:])
)


@jax.jit
def simulate_trajectory(model, init_data, actions):
    def rollout_step(current_data, action):
        current_data = current_data.replace(ctrl=action)

        def substep_fn(i, data):
            next_data = mjx.step(model, data)
            return jax.tree_util.tree_map(
                lambda n, c: (
                    n.astype(c.dtype) if hasattr(n, "dtype") else n
                ),
                next_data,
                data,
            )

        next_data = jax.lax.fori_loop(0, 10, substep_fn, current_data)
        next_data = jax.tree_util.tree_map(
            lambda n, c: n.astype(c.dtype) if hasattr(n, "dtype") else n,
            next_data,
            current_data,
        )
        return next_data, next_data

    _, trajectory_history = jax.lax.scan(
        rollout_step, init_data, actions
    )
    return trajectory_history


print("Simulating MJX trajectory...")
mjx_trajectory = simulate_trajectory(
    mjx_model, single_mjx_data, expert_actions
)
mjx_qpos_history = np.array(mjx_trajectory.qpos)

# 5. Render and Write Video
import imageio

video_path = "physics_divergence_comparison.mp4"
fps = 40  # 1 step = 0.025s -> 40 frames per second

print(f"Rendering and encoding web-safe video to {video_path}...")
num_frames = dataset_qpos.shape[0]

# Initialize imageio video writer with the highly compatible H.264 codec
with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as video_writer:
    for t in range(num_frames):
        # --- Render Dataset State (Left Panel) ---
        mj_data.qpos[:] = dataset_qpos[t]
        mj_data.qvel[:] = dataset_qvel[t]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frame_dataset = renderer.render()

        # --- Render MJX Rollout State (Right Panel) ---
        mj_data.qpos[:] = mjx_qpos_history[t]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frame_mjx = renderer.render()

        # Add text overlay labels to the frames (MuJoCo outputs RGB arrays)
        cv2.putText(frame_dataset, "Dataset (CPU)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_mjx, "MJX Rollout (GPU)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Combine side-by-side horizontally
        combined_frame = np.hstack((frame_dataset, frame_mjx))
        
        # Write directly (imageio expects RGB, so no BGR conversion needed!)
        video_writer.append_data(combined_frame)

print("Video compilation complete! You can now view this safely in your browser/VS Code.")