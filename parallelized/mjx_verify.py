import glob
import os
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Force float64 for exact dataset matching
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

expert_physics = demo["physics"]
expert_actions = demo["action"]

qpos_init = jnp.array(expert_physics[:-1, :9])
qvel_init = jnp.array(expert_physics[:-1, 9:])
actions = jnp.array(expert_actions[:-1, :])
qpos_target = jnp.array(expert_physics[1:, :9])
qvel_target = jnp.array(expert_physics[1:, 9:])

num_transitions = qpos_init.shape[0]

# Vectorize initial data states
batch_mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(num_transitions))
batch_mjx_data = batch_mjx_data.replace(
    qpos=qpos_init, qvel=qvel_init, ctrl=actions
)


# 3. Define Step Function with Static Argument
# 'action_repeat' is marked static so JAX can unroll/compile the loop properly
@jax.jit(static_argnames=['action_repeat'])
def verify_sweep_step(model, data, action_repeat):
    
    def dmc_env_step(model, init_data):
        def substep_fn(i, current_data):
            next_data = mjx.step(model, current_data)
            
            # Match the output types exactly to the input types to prevent
            # X64 integer promotion from breaking the fori_loop boundary.
            return jax.tree_util.tree_map(
                lambda n, c: n.astype(c.dtype) if hasattr(n, 'dtype') else n,
                next_data, 
                current_data
            )
        
        return jax.lax.fori_loop(0, action_repeat, substep_fn, init_data)

    return jax.vmap(dmc_env_step, in_axes=(None, 0))(model, data)


# 4. Run the Sweep
# Define the range of repeats you want to investigate
candidate_repeats = [1, 2, 4, 5, 8, 10, 12, 15, 20]

print(f"| {'Repeat':<8} | {'Pos MAE':<12} | {'Pos Max Err':<12} | {'Vel MAE':<12} |")
print("|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|")

for repeat in candidate_repeats:
    # Simulates one multi-substep transition for all 1,000 steps
    # Triggers a quick JIT compile only on the very first time it encounters a unique 'repeat' value
    simulated_output = verify_sweep_step(
        mjx_model, batch_mjx_data, action_repeat=repeat
    )
    jax.block_until_ready(simulated_output.qpos)

    # Compute errors
    qpos_errors = jnp.abs(simulated_output.qpos - qpos_target)
    qvel_errors = jnp.abs(simulated_output.qvel - qvel_target)

    mean_pos = jnp.mean(qpos_errors)
    max_pos = jnp.max(qpos_errors)
    mean_vel = jnp.mean(qvel_errors)

    print(
        f"| {repeat:<8} | {mean_pos:<12.2e} | {max_pos:<12.2e} | {mean_vel:<12.2e} |"
    )