from stable_baselines3.common.policies import ActorCriticPolicy, MlpExtractor
from agents.maskdp.legacy_rewards_rl import ActorR as Actor
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import wandb
import time
import csv

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
                verbose: int = 0,
                csv_name: str = 'ppo_train_log.csv'):
                
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.save_every_steps = save_every_steps
        self.use_wandb = use_wandb

        self.csv_path = snapshot_dir / csv_name

    def _init_callback(self):
        '''This is called just once at training start'''
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        file_exists = self.csv_path.exists()
        self._csv_file = self.csv_path.open("a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file,
            fieldnames=[
                "wall_time",
                "timesteps",
                "episode_return",
                "episode_length"])
        if not file_exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

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
                row = {"wall_time": time.time(),
                       "timesteps": int(self.num_timesteps),
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
            path = self.snapshot_dir / f"ppo_step_{self.num_timesteps}.zip"
            self.model.save(str(path))
            if self.verbose > 0:
                print(f"[PPO] Saved checkpoint to {path}")

        ## Save actor networks weights as MaskDP
        ## The just run the eval script on the checkpoints
        return True

    def _on_training_end(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None