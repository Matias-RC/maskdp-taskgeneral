import glob
import os
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Ensure 64-bit precision for exact trajectory comparison
jax.config.update("jax_enable_x64", True)

# 1. Setup Model
mj_model = mujoco.MjModel.from_xml_path("custom_dmc_tasks/walker.xml")
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# 2. Load Data
expert_data_path = "maskdp_data/maskdp_train/walker/expert/walker_walk/train"
npz_files = glob.glob(os.path.join(expert_data_path, "*.npz"))
demo = np.load(npz_files[0])

expert_physics = demo["physics"]  # (1001, 18)
expert_actions = demo["action"]  # (1001, 6)
print(expert_actions)
# Slice data for the rollout
# We initialize at step 0
qpos_init = jnp.array(expert_physics[0, :9])
qvel_init = jnp.array(expert_physics[0, 9:])

# Actions to apply sequentially (steps 0 to 999)
actions_sequence = jnp.array(expert_actions[:-1, :], dtype=jnp.float64)

# The true trajectory we are trying to match (steps 1 to 1000)
qpos_targets = jnp.array(expert_physics[1:, :9])
qvel_targets = jnp.array(expert_physics[1:, 9:])

# Initialize our single trajectory state
single_mjx_data = mjx_data.replace(qpos=qpos_init, qvel=qvel_init)

# 3. Define the Sequential Rollout Function using lax.scan
@jax.jit
def simulate_trajectory(model, init_data, actions):

    def rollout_step(current_data, action):
        # Apply the current action (now safely float64)
        current_data = current_data.replace(ctrl=action)

        # Run 10 physics substeps
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

        # Safety Check: Ensure the final output carry matches the exact types 
        # that current_data had when it entered this specific step.
        next_data = jax.tree_util.tree_map(
            lambda n, c: n.astype(c.dtype) if hasattr(n, "dtype") else n,
            next_data,
            current_data
        )

        return next_data, next_data

    _, trajectory_history = jax.lax.scan(
        rollout_step, init_data, actions
    )
    return trajectory_history

# 4. Run the Rollout
print("Compiling and running sequential 1000-step rollout...")
trajectory = simulate_trajectory(
    mjx_model, single_mjx_data, actions_sequence
)
jax.block_until_ready(trajectory.qpos)

# 5. Calculate Compounding Errors over Time
# Calculate the Mean Absolute Error (MAE) at each individual timestep
step_pos_mae = jnp.mean(
    jnp.abs(trajectory.qpos - qpos_targets), axis=-1
)
step_vel_mae = jnp.mean(
    jnp.abs(trajectory.qvel - qvel_targets), axis=-1
)

# Calculate Euclidean distance (L2 norm) for joint positions at each step
# This gives a clear physical sense of how far the walker has drifted
position_drift_l2 = jnp.linalg.norm(
    trajectory.qpos - qpos_targets, axis=-1
)

# 6. Print the Compounding Progress Report
print("\n" + "=" * 65)
print("             COMPOUNDING TRAJECTORY DRIFT REPORT            ")
print("=" * 65)
print(
    f"| {'Env Step':<10} | {'Sim Time (s)':<14} | {'Pos MAE':<12} | {'L2 Drift (m)':<14} |"
)
print(
    "|"
    + "-" * 12
    + "|"
    + "-" * 16
    + "|"
    + "-" * 14
    + "|"
    + "-" * 16
    + "|"
    + "-" * 14
    + "|"
)

# Print milestones throughout the 25-second episode
milestones = [0, 9, 49, 99, 249, 499, 749, 999]
for idx in milestones:
    sim_time = (idx + 1) * 0.025  # 10 substeps * 0.0025s per step
    print(
        f"| {idx+1:<10} | {sim_time:<14.3f} | {step_pos_mae[idx]:<12.4f} | {position_drift_l2[idx]:<14.4f} |"
    )
print("=" * 65)