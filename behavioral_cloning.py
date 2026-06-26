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
from logger import Logger
from replay_buffer import make_replay_loader

from expert_finetuning.bc_policy import MaskDP_BC 

torch.backends.cudnn.benchmark = True

def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]

@hydra.main(config_path=".", config_name="behavioral_cloning")
def main(cfg):
    work_dir = Path.cwd()
    print(f"Workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    env = dmc.make(cfg.task, seed=cfg.seed)
    obs_shape = env.observation_spec().shape
    action_shape = env.action_spec().shape

    print(f"Initializing BC Agent and loading pretrained weights from: {cfg.pretrained_path}")
    agent = MaskDP_BC(
        cfg=cfg.agent, 
        obs_shape=obs_shape[0],
        action_shape=action_shape[0],
        device=device,
        lr=cfg.lr,
        use_tb=cfg.use_tb,
        path=cfg.pretrained_path
    ).to(device)

    # 4. Set up Logging and Directories
    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)
    
    exp_name = "_".join(["BC", domain, str(cfg.seed)])
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
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # 5. Initialize Offline Expert Data Loader
    replay_train_dir = Path(cfg.replay_buffer_dir) / domain
    print("Using expert dataset:", replay_train_dir)
    train_loader = make_replay_loader(
        env,
        replay_train_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        domain,
        cfg.agent.transformer_cfg.traj_length,
        relabel=False,
        is_local_data=cfg.get("is_local_data", True),
        hf_path=cfg.get("hf_path", None)
    )
    train_iter = iter(train_loader)

    # 6. Training Loop
    timer = utils.Timer()
    global_step = cfg.get("resume_step", 0)

    train_until_step = utils.Until(cfg.num_grad_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    print(f"Starting Behavioral Cloning for {cfg.num_grad_steps} steps...")
    
    while train_until_step(global_step):
        batch = next(train_iter)
        obs, action, reward, discount, next_obs, step_type = utils.to_torch(batch, device)

        metrics = agent.rex_requested_act(obs, action)

        logger.log_metrics(metrics, global_step, ty="train")
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        if hasattr(cfg, 'snapshots') and global_step in cfg.snapshots:
            snapshot = snapshot_dir / f"snapshot_{global_step}.pt"
            payload = {
                "model": agent.backbone.state_dict(),
                "cfg": cfg.agent,
            }
            with snapshot.open("wb") as f:
                torch.save(payload, f)

        global_step += 1

    print("Behavioral Cloning complete!")
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()