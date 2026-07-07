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
from utils.video import VideoRecorder
from omegaconf import DictConfig, OmegaConf
from data.replay_buffer import make_replay_loader
from utils.utils import set_seed_everywhere, Until, Every, Timer

# Everything eval-specific (get_domain, get_dir, eval_mdp, eval_bc, eval_seq_bc,
# eval_dataset, ...) now lives in utils/eval_utils.py, so this root file stays
# limited to `main`, same as offline.py / online.py.
from utils.eval_utils import (
    get_domain,
    get_dir,
    eval_seq_bc,
    eval_bc,
    eval_mdp,
    eval_dataset,
)


@hydra.main(version_base=None, config_path="configs", config_name="eval_offline")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    # Print the actual directory determined by hydra config
    print(f"Workspace: {work_dir}")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    # Set seed for random, numpy and torch
    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Since we may work with dmc, mtm_dmc or future environment standards
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/
    env_cls = instantiate(cfg.env_cls_route)

    # Create environment for the task the snapshot was trained on
    env = env_cls.make(cfg.task, seed=cfg.seed)

    # Load the trained agent from its snapshot
    path = get_dir(cfg)
    agent = instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
        path=path,
    )

    # Create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    cfg.agent.transformer_cfg = agent.config
    exp_name = "_".join([cfg.agent.name, cfg.task, str(cfg.replan), str(cfg.seed)])
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
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # Create data storage
    domain = get_domain(cfg.task)
    goal_dir = Path(cfg.goal_buffer_dir) / cfg.task
    print(f"goal buffer dir: {goal_dir}")

    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)
    global_step = 0
    timer = Timer()

    # --- Dataset Inspection Route ---
    if getattr(cfg, "inspect_data_only", False):
        import copy

        # 1. Deepcopy agent config and unlock it to allow custom fields
        buffer_cfg = copy.deepcopy(agent.config)
        OmegaConf.set_struct(buffer_cfg, False)

        # 2. Define lengths for the dataset tracking
        buffer_cfg.context_length = 250  # How many sequential action steps you want to evaluate
        buffer_cfg.forecast_length = 1   # Dummy length to prevent slice math errors in the legacy code

        dataset_loader = make_replay_loader(
            env,
            goal_dir,
            cfg.goal_buffer_size,
            cfg.num_eval_episodes,
            cfg.goal_buffer_num_workers,
            cfg.discount,
            domain=domain,
            mode="prompt",
            cfg=buffer_cfg,  # Pass the patched configuration here
            relabel=False,
        )
        dataset_iter = iter(dataset_loader)
        eval_dataset(
            global_step=global_step,
            agent=agent,
            env=env,
            logger=logger,
            dataset_iter=dataset_iter,
            device=device,
            num_eval_episodes=cfg.num_eval_episodes,
            video_recorder=video_recorder,
            cfg=cfg,
        )
        print("Done!")
        return

    # --- Standard Model Evaluation Route ---
    goal_loader = make_replay_loader(
        env,
        goal_dir,
        cfg.goal_buffer_size,
        cfg.num_eval_episodes,
        cfg.goal_buffer_num_workers,
        cfg.discount,
        domain=domain,
        traj_length=1,
        mode="goal",
        cfg=agent.config,
        relabel=False,
    )
    goal_iter = iter(goal_loader)

    EVAL_DISPATCH = {
        "mdp_goal": lambda **kw: eval_mdp(**kw, replan=cfg.replan),
        "bc_goal": eval_bc,
        "seq_goal": eval_seq_bc, # Add any missing functions here
    }

    eval_fn = EVAL_DISPATCH.get(cfg.agent.name)
    if eval_fn is None:
        raise NotImplementedError(f"No eval function for agent {cfg.agent.name}")
    eval_fn(global_step, agent, env, logger, goal_iter, device, cfg.num_eval_episodes, video_recorder)

    print("Done!")


if __name__ == "__main__":
    main()