from data.online_rollout_buffer import CustomPPO_RB
from agents.policies.mdpActorCustomPPO import MDPWrapperPPO
from torch.distributions import Normal
from utils.logger import AverageMeter
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch as th
import gym


class PPOTrainer:
    rb: CustomPPO_RB
    env: gym.vector.AsyncVectorEnv
    agent: MDPWrapperPPO
    def __init__(
            self, 
            env, 
            agent,
            device,
            cfg,
            seed=1,
        ):
        self.env = env
        self.num_epochs = cfg.num_epochs
        self.num_workers = env.num_envs
        self.device = device
        self.rb = CustomPPO_RB(
            size=(cfg.rb_size, int(env.get_attr("_max_episode_steps")[0])), 
            actions_shape=env.action_space.shape,
            states_shape=env.observation_space.shape,
            reward_shape=(env.num_envs, 1,),
            num_workers=env.num_envs,
            context_window=agent.model.traj_length-1,
            batch_size=cfg.batch_size,
            device=device
        )
        self.agent = agent

    def fix_input(self, obs, reward, dones, truncated, info):
        actual_obs = th.tensor(np.stack(info["obs2"]))
        info["obs2"] = obs
        return actual_obs, reward, dones | truncated, info
    
    def update(self, global_step):
        obs = self.env.reset()
        obs = th.from_numpy(obs[0]).to(self.device)
        self.rb.reset()
        for i in range(self.num_workers):
            self.rb.set_initial_obs(obs[i], i)
        dist, Vf = self.agent.act(*self.rb.fetch())
        while not self.rb.full:
            self.rb.step()
            a = dist.sample()
            log_prob = dist.log_prob(a).sum(dim=-1)
            obs, reward, terminated, truncated, info = self.env.step(a.detach().cpu().numpy())
            obs, reward, dones, info = self.fix_input(obs, reward, terminated, truncated, info)
            self.rb.append(a, Vf, log_prob, obs, reward, info)
            dist, Vf = self.agent.act(*self.rb.fetch())
            for i in range(self.num_workers):
                if not dones[i]: continue
                if self.rb.full: break
                self.rb.add(i, Vf[i])
                new_obs = th.from_numpy(info["obs2"][i]).to(self.device)
                self.rb.set_initial_obs(new_obs, i)
        rew = self.rb.get_avg_reward_trace()
        train_loader = self.rb.to_static()
        
        metrics_writer = {"vf_head": defaultdict(AverageMeter), "pi_head": defaultdict(AverageMeter)}
        for _ in range(self.num_epochs):
            for batch in train_loader:
                value_head_metrics = self.agent.update_value_head(batch, global_step)
                policy_head_metrics = self.agent.update_policy(batch, global_step)
                for key, value in value_head_metrics.items():
                    metrics_writer["vf_head"][key].update(value)
                for key, value in policy_head_metrics.items():
                    metrics_writer["pi_head"][key].update(value)
        metrics = {"avg_reward_trace": rew}
        for ikey, ivalue in metrics_writer.items():
            for jkey, jvalue in ivalue.items():
                metrics[f"{ikey}_{jkey}"] = jvalue.value()
        return metrics


                

