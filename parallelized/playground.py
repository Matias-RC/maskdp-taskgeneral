import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import time

print(f"Device: {jax.devices()} | Backend: {jax.default_backend()}")

# 1. Load model
mj_model = mujoco.MjModel.from_xml_path("custom_dmc_tasks/walker.xml")
mj_data = mujoco.MjData(mj_model)

# 2. Put onto GPU
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# 3. Vectorize to 4,096 envs
batch_size = 2048
batch_mjx_data = jax.vmap(lambda _: mjx_data)(jnp.arange(batch_size))

# 4. Define parallel step
@jax.jit
def parallel_step(model, data, actions):
    # Apply the actions to the motor actuators
    data = data.replace(ctrl=actions)
    return jax.vmap(mjx.step, in_axes=(None, 0))(model, data)

# --- 5. THE WARMUP (Compiles the function) ---
print("\nCompiling JIT function (this takes a moment)...")
# Generate dummy actions for your 6 actuators (hip, knee, ankle for 2 legs)
dummy_actions = jnp.zeros((batch_size, 6)) 
start_compile = time.time()
batch_mjx_data = parallel_step(mjx_model, batch_mjx_data, dummy_actions)
jax.block_until_ready(batch_mjx_data.qpos)
print(f"Compilation finished in {time.time() - start_compile:.2f} seconds.")

# --- 6. THE ACTUAL SIMULATION LOOP ---
steps = 500
print(f"\nRunning {steps} steps across {batch_size} parallel environments...")

key = jax.random.PRNGKey(0)
start_sim = time.time()

for _ in range(steps):
    # Split random key to generate random motor actions between -1.0 and 1.0
    key, subkey = jax.random.split(key)
    random_actions = jax.random.uniform(subkey, shape=(batch_size, 6), minval=-1.0, maxval=1.0)
    
    # Step all 4,096 environments at once
    batch_mjx_data = parallel_step(mjx_model, batch_mjx_data, random_actions)

# Wait for the GPU to finish the last step before stopping the clock
jax.block_until_ready(batch_mjx_data.qpos)
end_sim = time.time()

# Calculate Performance
total_time = end_sim - start_sim
total_steps = batch_size * steps
sps = total_steps / total_time

print("--- RESULTS ---")
print(f"Total Simulation Time: {total_time:.4f} seconds")
print(f"Total Steps Simulated: {total_steps:,}")
print(f"Steps Per Second (SPS): {sps:,.0f}!")