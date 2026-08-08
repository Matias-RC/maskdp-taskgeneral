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

####
from utils import video
torch.backends.cudnn.benchmark = True

@hydra.main(config_path="configs", config_name="eval_online")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    # Print the actual directory determined by hydra config
    print(f"Workspace: {work_dir}")

    # Set seed for random, numpy and torch
    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Since we may work with dmc, mtm_dmc or future environment standards
    env_maker = instantiate(
        cfg.env_cls_route
    )

    # Create environment for specified task
    env = env_maker(cfg.task, cfg.num_workers, seed=cfg.seed)
    agent = instantiate(
        cfg.agent,
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
    )
    evaluator = instantiate(
        cfg.trainer, #Activates configs from "Algorithms" and consists of an eval from partial PPO rollout
        env=env,
        agent=agent,
        device=device
    )

    # Create logger
    cfg.agent.obs_shape = env.observation_space.shape
    cfg.agent.action_shape = env.action_space.shape
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

    timer = Timer()

    ######
    video_recorder = video.VideoRecorder(root_dir=work_dir, fps=20, render_size=256)

    global_step = cfg.resume_step

    train_until_step = Until(cfg.num_grad_steps)
    #eval_every_step = Every(cfg.eval_every_steps)
    log_every_step = Every(cfg.log_every_steps)
    # True until global_step gets to cfg.num_grad_steps
    while train_until_step(global_step):
        ####
        recorder_to_pass = video_recorder if cfg.save_video else None

        avg_rew = evaluator.roll(global_step, video_recorder=recorder_to_pass)

        ####
        if cfg.save_video:
            video_name = f"eval_step_{global_step}.mp4"
            video_recorder.save(video_name)
            print(f"¡Video guardado con éxito!: {video_name}")

        metrics = {"avg_rew":avg_rew}
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
if __name__ == "__main__":
    main()
