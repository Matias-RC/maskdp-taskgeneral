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


class OnlineEvaluator:
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

    def roll(self, global_step, video_recorder=None):
        obs = self.env.reset()
        obs = th.from_numpy(obs[0]).to(self.device)
        self.rb.reset()
        for i in range(self.num_workers):
            self.rb.set_initial_obs(obs[i], i)
        dist, Vf = self.agent.act(*self.rb.fetch())
        
        ####
        if video_recorder is not None:
            video_recorder.init(self.env.envs[0], enabled=True) #entorno del worker 0
            step_count = 0

        while not self.rb.full:
            self.rb.step()
            a = dist.sample()
            log_prob = dist.log_prob(a).sum(dim=-1)
            obs, reward, terminated, truncated, info = self.env.step(a.detach().cpu().numpy())
            obs, reward, dones, info = self.fix_input(obs, reward, terminated, truncated, info)
            
            ####
            if video_recorder is not None:
                metadata = {
                    "step": step_count, 
                    "rew": round(float(reward[0]), 2)  # Recompensa del worker 0
                }
                video_recorder.record(self.env.envs[0], metadata=metadata)
                step_count += 1
                
                if dones[0]:
                    video_recorder.enabled = False

            self.rb.append(a, Vf, log_prob, obs, reward, info)
            dist, Vf = self.agent.act(*self.rb.fetch())
            for i in range(self.num_workers):
                if not dones[i]: continue
                if self.rb.full: break
                self.rb.add(i, Vf[i])
                new_obs = th.from_numpy(info["obs2"][i]).to(self.device)
                self.rb.set_initial_obs(new_obs, i)
                
        return self.rb.get_avg_reward_trace()