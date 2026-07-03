import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "disable"

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn

import dmc
# Having this line before import dmc won't work
# I think import numpy resets warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import utils
from stable_baselines3 import PPO
from dmc_to_gym import DMCGymWrapper
from stable_baselines3.common.monitor import Monitor # This wraps gym envs
# This allows to make a custom callback class to collect training data
from stable_baselines3.common.callbacks import BaseCallback
# This is for the custom policy using MaskDP as an actor
from stable_baselines3.common.policies import ActorCriticPolicy, MlpExtractor
from stable_baselines3.common.env_checker import check_env

# Actor expects observation sequence of size (batch, timesteps, dimension)
from agent.mdp_rl import Actor

# For logging
import wandb
import omegaconf

torch.backends.cudnn.benchmark = True

class MaskDPActorPPOPolicy(ActorCriticPolicy):
    """
    PPO policy that uses default feature extractor and value network from SB3,
    but replaces the actor with a pre-trained MaskDP Actor.
    """

    def __init__(self, *args,
                 maskdp_actor: Actor, fixed_std: float=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskdp_actor = maskdp_actor
        self.fixed_std = fixed_std

    def _build_mlp_extractor(self):
        # 𝜋 - Identity policy network beacuse we're using MaskDP
        # V - Two-layer 64-neuron network, as is standard for actor-critic
        net_arch = [dict(pi=[], vf=[64, 64])]
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=net_arch,
            activation_fn=self.activation_fn)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):

        # (batch, state_dim) 
        # barch is always 1 here though
        state = latent_pi.unsqueeze(1)  # (batch, 1, state_dim)

        # MaskDP Actor returns a distribution
        # (batch, 1, action_dim)
        dist = self.maskdp_actor(state, std=self.fixed_std)
        mean_actions = dist.mean[:, -1, :]  # (batch, action_dim)

        # return SB3's Gaussian distribution
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

class SaveAndLogCallback(BaseCallback):
    def __init__(self,
                snapshot_dir: Path,
                save_every_steps: int,
                use_wandb: bool = False,
                verbose: int = 0):
                
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.save_every_steps = save_every_steps
        self.use_wandb = use_wandb

    def _on_step(self) -> bool:
        info = self.locals.get("infos")[0]
        # This is wen the step finished an episode
        if "episode" in info:
            print("INFO")
            print(info)
            raise
            ep = info["episode"]
            ep_reward = ep["r"]
            ep_len = ep["l"]

            if self.verbose > 0:
                print(f"[PPO] ep_reward={ep_reward:.2f}, len={ep_len}")

            if self.use_wandb:
                wandb.log(
                    {"train/episode_return": ep_reward,
                     "train/episode_length": ep_len,
                     "train/num_timesteps": self.num_timesteps},
                    step=self.num_timesteps,
                )
        else:
            # print("No episode in info")
            pass

        # Save checkpoints
        if self.num_timesteps % self.save_every_steps == 0:
            path = self.snapshot_dir / f"ppo_step_{self.num_timesteps}.zip"
            self.model.save(str(path))
            if self.verbose > 0:
                print(f"[PPO] Saved checkpoint to {path}")

        ## Save actor networks weights as MaskDP
        ## The just run the eval script on the checkpoints

        return True



def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / f"snapshot_{cfg.resume_step}.pt"
    print("loading from", snapshot)
    return snapshot

def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]

# This links it to eval.yaml
@hydra.main(config_path=".", config_name="finetune_ppo")
def main(cfg):

    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Create a snapshot directory for the task
    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / \
                   domain / str(cfg.seed) / cfg.algorithm
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # Create dmc env and convert to gym (Not gymnasium)
    dmc_env = dmc.make(cfg.task, seed=cfg.seed)
    env = DMCGymWrapper(dmc_env)
    check_env(env) # This verifies env actually complies
    env = Monitor(env) # Wraps for SB3

    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=dmc_env.observation_spec().shape,
        action_shape=dmc_env.action_spec().shape)

    if cfg.resume:
        pretrained = torch.load(get_dir(cfg), map_location=cfg.device)
        agent.model.load_state_dict(pretrained["model"], strict=True)

    # === Build MaskDP Actor for PPO ===
    obs_dim = env.observation_space.shape[0] # 24
    action_dim = env.action_space.shape[0]   # 6

    # Number of tokens is (2 * trajectory length) because each step in
    # the trajectory has one token for state and one for action
    attn_len = cfg.agent.transformer_cfg.traj_length * 2
    # finetune doesn't actually get used, so it can be whatever
    maskdp_actor = Actor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        attention_length=attn_len,
        finetune='whatever',
        config=cfg.agent.transformer_cfg,
    ).to(device)

    # Load weights
    maskdp_actor.mdp.load_state_dict(agent.model.state_dict())

    policy_kwargs = dict(
        maskdp_actor=maskdp_actor,
        fixed_std=cfg.fixed_std if "fixed_std" in cfg else 0.1)

    model = PPO(
        policy=MaskDPActorPPOPolicy,
        env=env,
        device=device,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        verbose=1)

    # Callback for logging + checkpoints
    callback = SaveAndLogCallback(
        snapshot_dir=snapshot_dir,
        save_every_steps=cfg.log_every_steps,
        use_wandb=getattr(cfg, "use_wandb", False),
        verbose=1)

    model.learn(total_timesteps=10, #cfg.num_grad_steps
                callback=callback) 

if __name__ == "__main__":
    main()