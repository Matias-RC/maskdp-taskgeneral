import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# https://gymnasium.farama.org/environments/mujoco/
os.environ["MUJOCO_GL"] = "disable" # egl doesn't work on Peteroa :(

from pathlib import Path

import hydra
import torch
import wandb
import omegaconf
from pprint import pprint
from utils.logger import Logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from data.replay_buffer import make_replay_loader
from utils.utils import set_seed_everywhere, Until, Every, Timer

torch.backends.cudnn.benchmark = True


#Offline training
# https://hydra.cc/docs/tutorials/basic/your_first_app/using_config/
# An introduction to the use of the hydra config decorator
@hydra.main(version_base=None, config_path="configs", config_name="offline")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    # Print the actual directory determined by hydra config
    print(f"Workspace: {work_dir}")

    # Set seed for random, numpy and torch
    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Since we may work with dmc, mtm_dmc or future environment standards
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/
    # A brief explanation to object instantiation with hydra
    env_cls = instantiate(
        cfg.env_cls_route
    )

    # Create environment for specified task
    env = env_cls.make(cfg.task, seed=cfg.seed)

    # Create agent. Utils will instantiate a class
    # Since cfg.agent is 'mdp'. The class will be specified by configs/agent/*.yaml
    # in the agent folder
    agent = instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
    )
    
    # Create snapshot directory
    # cfg.context : Since both behavioural cloning and pretraining are offline the script is reused for both.
    snapshot_dir = work_dir / "snapshots" / cfg.context / cfg.domain / cfg.algorithm / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # Create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    exp_name = "_".join([
        cfg.agent.name, cfg.domain, str(cfg.seed), str(cfg.algorithm)
    ])
    # Create wandb_config from Hydra's omegaconf
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.project,
        # This has to be your WandB user-institution
        entity=None, 
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(_disable_stats=True,),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # Initialize Offline Data Loader
    replay_train_dir = Path(cfg.replay_buffer_dir)
    print("Using dataset:", replay_train_dir)
    train_loader = make_replay_loader(
        env=env,
        replay_dir=replay_train_dir,
        max_size=cfg.replay_buffer_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.replay_buffer_num_workers,
        discount=cfg.discount,
        traj_length=cfg.agent.transformer_cfg.traj_length,
        relabel=False,
        is_local_data=cfg.is_local_data,
        hf_path=cfg.hf_path
    )
    train_iter = iter(train_loader)

    timer = Timer()

    global_step = cfg.resume_step

    train_until_step = Until(cfg.num_grad_steps)
    eval_every_step = Every(cfg.eval_every_steps)
    log_every_step = Every(cfg.log_every_steps)

    # True until global_step gets to cfg.num_grad_steps
    while train_until_step(global_step):
        # try to evaluate
        # Train on a single batch and permform a gradient step
        metrics = agent.update(train_iter, global_step)
        # Log each metric using the "Train meter group" on the logger
        logger.log_metrics(metrics, global_step, ty="train")
        # Log just registers the metrics on a MetersGroup instance
        # inside the logger
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)
            # Upon exiting the context manager "LogAndDumpCtx", the logged
            # data is actually dumped to WandB

        if global_step in cfg.snapshots:
            snapshot = snapshot_dir / f"snapshot_{global_step}.pt"
            payload = {
                "model": agent.model.state_dict(),
                "cfg": cfg.agent.transformer_cfg,
            }
            with snapshot.open("wb") as f:
                torch.save(payload, f)

        global_step += 1
    print("Done!")


if __name__ == "__main__":
    main()
