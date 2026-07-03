import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "disable"

from pathlib import Path
import hydra
import torch
import wandb
import omegaconf

import dmc
import utils

# Assuming these are accessible in your workspace:
from ppo.actorCritic import MaskDPA2C 
from ppo.engine import MaskDPTrainer
from ppo.sequenceRolloutBuffer import SequenceRB
from ppo.callback import WandBSnapshotCallback

torch.backends.cudnn.benchmark = True

def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]

def get_max_steps(env):
    # unwrap
    while hasattr(env, "_env"):
        env = env._env

    # New style: step limit already stored
    if hasattr(env, "_step_limit"):
        return int(env._step_limit)

    # Old style: convert seconds → steps
    time_limit = getattr(env, "_time_limit", None)

    if time_limit is not None:
        control_step = env._task.control_timestep()
        return int(round(time_limit / control_step))

    raise ValueError("Could not determine episode length")

@hydra.main(config_path=".", config_name="finetune_A2C")
def main(cfg):
    work_dir = Path.cwd()
    print(f"Workspace: {work_dir}")

    # Set up seeds and device
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # 1. Create DeepMind Control environment
    env = dmc.make(cfg.task, seed=cfg.seed)
    obs_shape = env.observation_spec().shape
    action_shape = env.action_spec().shape

    # 2. Initialize the Agent (MaskDPA2C)
    print(f"Initializing Agent and loading pretrained weights from: {cfg.pretrained_path}")
    agent = MaskDPA2C(
        cfg=cfg.agent, 
        obs_shape=obs_shape[0],
        action_shape=action_shape[0],
        device=device,
        lr=cfg.lr,
        use_tb=cfg.use_tb,
        path=cfg.pretrained_path
    ).to(device)

    max_steps = get_max_steps(env)
    # 3. Initialize the Sequence Replay Buffer
    buffer = SequenceRB(
        size=cfg.buffer_size,
        s_shape=obs_shape[0],
        a_shape=action_shape[0],
        timeout_limit=max_steps,
        device=device
    )

    # 4. Set up Logging and Callbacks
    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    
    exp_name = "_".join(["PPO", domain, str(cfg.seed)])
    with omegaconf.open_dict(cfg):
        cfg.agent.obs_shape = obs_shape[0]
        cfg.agent.action_shape = action_shape[0]
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    if cfg.use_wandb:
        wandb.init(
            project=cfg.project,
            name=exp_name,
            config=wandb_config,
            settings=wandb.Settings(_disable_stats=True),
            mode="online",
            notes=cfg.notes,
        )

    callback = WandBSnapshotCallback(
        snapshot_dir=str(snapshot_dir),
        save_every=cfg.save_every,
        run_name=exp_name,
        config=wandb_config if cfg.use_wandb else None
    )

    # 5. Initialize the Trainer and start the loop
    trainer = MaskDPTrainer(
        agent=agent,
        env=env,
        buffer=buffer,
        device=device,
        cfg=cfg
    )

    print(f"Starting PPO Fine-Tuning for {cfg.iterations} iterations...")
    trainer.train(
        iterations=cfg.iterations, 
        epochs=cfg.epochs, 
        callback=callback
    )
    
    print("Fine-tuning sequence complete!")
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()