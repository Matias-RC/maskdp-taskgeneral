import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "disable"

from pathlib import Path

import csv
import time

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
from agent.mdpr_rl import ActorR, CriticR

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
                 maskdp_actor: ActorR, context_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskdp_actor = maskdp_actor
        self.context_len = context_length

    def _build_mlp_extractor(self):
        # 𝜋 - Identity policy network beacuse we're using MaskDP
        # V - Two-layer 64-neuron network, as is standard for actor-critic
        net_arch = [dict(pi=[], vf=[64, 64])]
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=net_arch,
            activation_fn=self.activation_fn)

    def _get_action_dist_from_latent(self, latent_pi):
        # print("Using actor:", self.maskdp_actor.__class__.__name__)

        # latent_pi comes in (B, T * state_dim)
        B, flat_dim = latent_pi.shape
        T = self.context_len
        state_dim = flat_dim // T
        obs_seq = latent_pi.reshape(B, T, state_dim)
        # print("obs_seq:", obs_seq.shape)

        mean_action = self.maskdp_actor(obs_seq) # (B, 3T, embd_dim)
        # print("Pred act shape:", ret.shape)

        # return SB3's Gaussian distribution
        return self.action_dist.proba_distribution(mean_action, self.log_std)

class MaskDPActorCriticPPOPolicy(ActorCriticPolicy):
    """
    PPO policy that replaces the actor and critic with a pre-trained MaskDP model.
    """
    def __init__(self, *args,
                 maskdp_actor: ActorR, maskdp_critic: CriticR,
                 context_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskdp_actor = maskdp_actor
        self.maskdp_critic = maskdp_critic
        self.context_len = context_length

    def _untangle_sequence(self, latent):
        # latent comes in (B, T * state_dim) -> (B, T, state_dim)
        B, flat_dim = latent.shape
        T = self.context_len
        state_dim = flat_dim // T
        return latent.reshape(B, T, state_dim)

    def _build_mlp_extractor(self):
        # 𝜋 - Identity policy network beacuse we're using MaskDP
        # V - Two-layer 64-neuron network, as is standard for actor-critic
        net_arch = [dict(pi=[], vf=[64, 64])]
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=net_arch,
            activation_fn=self.activation_fn)

    def _get_action_dist_from_latent(self, obs_seq):
        mean_action = self.maskdp_actor(obs_seq) # (B, action_dim)
        # print("Pred act shape:", ret.shape)
        return self.action_dist.proba_distribution(mean_action, self.log_std)

    
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs)
        obs_seq = self._untangle_sequence(features)
        values = self.maskdp_critic(obs_seq)                      # Critic
        distribution = self._get_action_dist_from_latent(obs_seq) # Actor

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions: torch.Tensor):
        features = self.extract_features(obs)
        obs_seq = self._untangle_sequence(features)

        distribution = self._get_action_dist_from_latent(obs_seq)
        log_prob = distribution.log_prob(actions)
        values = self.maskdp_critic(obs_seq)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        print("Predicting values...")
        print("OBS:", obs.shape)
        features = self.extract_features(obs)
        print("features", features.shape)
        obs_seq = self._untangle_sequence(features)
        return self.maskdp_critic(obs_seq)
        
"SRC: https://stable-baselines3.readthedocs.io/en/master/"
"_modules/stable_baselines3/common/policies.html#ActorCriticPolicy"

class SaveAndLogCallback(BaseCallback):
    def __init__(self,
                save_every_steps: int,
                use_wandb: bool = False,
                verbose: int = 0,
                algorithm: str = 'scratch'):
                
        super().__init__(verbose)
        self.save_every_steps = save_every_steps
        self.use_wandb = use_wandb
        self.algorithm = algorithm

        self.csv_path = Path(f"./ppo_train_log_{algorithm}.csv")
        self.ckpt_dir = Path(f"./ppo_checkpoints_{algorithm}")   

    def _init_callback(self):
        '''This is called just once at training start'''
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file,
            fieldnames=[
                "timesteps",
                "episode_return",
                "episode_length"])
        self._csv_writer.writeheader()
        self._csv_file.flush()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]

            # This is when the step finished an episode
            if "episode" in info:
                ep = info["episode"]
                ep_reward = float(ep["r"])
                ep_len = int(ep["l"])

                # --- CSV log ---
                row = {"timesteps": int(self.num_timesteps),
                       "episode_return": ep_reward,
                       "episode_length": ep_len}
                self._csv_writer.writerow(row)
                self._csv_file.flush()

                if self.verbose > 0:
                    print(f"[PPO] ep_reward={ep_reward:.2f}, len={ep_len}")

                if self.use_wandb:
                    wandb.log(
                        {"train/episode_return": ep_reward,
                        "train/episode_length": ep_len,
                        "train/num_timesteps": self.num_timesteps},
                        step=self.num_timesteps)

        # Save checkpoints
        if self.num_timesteps % self.save_every_steps == 0:
            # PPO checkpoint
            # ppo_path = self.ckpt_dir / \
            #     f"{self.algorithm}_ppo_step_{self.num_timesteps}.zip"
            # self.model.save(str(ppo_path))

            # MaskDP actor checkpoint
            actor_path = self.ckpt_dir / \
                f"{self.algorithm}_actor_step_{self.num_timesteps}.pt"
            torch.save(self.model.policy.maskdp_actor.state_dict(), actor_path)

        return True

    def _on_training_end(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None


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
    env = DMCGymWrapper(dmc_env, obs_seq_len=cfg.agent.transformer_cfg.traj_length)
    check_env(env) # This verifies env actually complies
    env = Monitor(env) # Wraps for SB3

    print("Obs Shape for agent:", dmc_env.observation_spec().shape)
    print("Act Shape for agent:", dmc_env.action_spec().shape)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=dmc_env.observation_spec().shape,
        action_shape=dmc_env.action_spec().shape)

    if cfg.resume:
        pretrained = torch.load(get_dir(cfg), map_location=cfg.device)
        missing, unexpected = agent.model.load_state_dict(pretrained["model"], strict=False)
        print("[load_state_dict] missing keys:", missing)
        print("[load_state_dict] unexpected keys:", unexpected)

    # === Build MaskDP Actor for PPO ===
    obs_dim = env.observation_space.shape # (T, latent_size)
    action_dim = env.action_space.shape   # (num_actions,)
    
    print("Obs dim:", obs_dim)
    print("Act dim:", action_dim)

    # # Number of tokens is (3 * trajectory length) because each step in
    # # the trajectory has one token for state, one for action and one for reward
    attn_len = cfg.agent.transformer_cfg.traj_length * 3
    maskdp_actor = ActorR(
        obs_dim[-1], action_dim[-1],
        attn_len,
        cfg.agent.transformer_cfg
    ).to(device)

    maskdp_critic = CriticR(
        obs_dim[-1], action_dim[-1],
        attn_len,
        cfg.agent.transformer_cfg
    ).to(device)

    # Load weights
    maskdp_actor.model.load_state_dict(agent.model.state_dict())
    maskdp_critic.model.load_state_dict(agent.model.state_dict())

    policy_kwargs = dict(
        maskdp_actor=maskdp_actor,
        maskdp_critic=maskdp_critic,
        context_length=cfg.agent.transformer_cfg.traj_length)

    model = PPO(
        policy=MaskDPActorCriticPPOPolicy,
        env=env,
        device=device,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        verbose=1)

    # Callback for logging + checkpoints
    callback = SaveAndLogCallback(
        save_every_steps=cfg.log_every_steps,
        use_wandb=cfg.use_wandb,
        verbose=1,
        algorithm=cfg.algorithm)

    model.learn(total_timesteps=cfg.num_grad_steps,
                callback=callback) 

if __name__ == "__main__":
    main()