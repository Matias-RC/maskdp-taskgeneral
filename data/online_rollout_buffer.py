from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch as th
import numpy as np
import math

class CustomPPO_RB:
    initial_obs: th.Tensor
    def __init__(self, size, actions_shape, states_shape, reward_shape, num_workers, context_window, batch_size, device ,gamma=0.99, lam=0.95):
        self.device = device
        self.size = size # tuple (B,L)
        self.lengths = th.zeros((size[0],), device=device)
        self.states = th.zeros((size[0], size[1]+1,)+(states_shape[1],), device=device)
        self.actions = th.zeros(size+(actions_shape[1],), device=device)
        self.log_probs = th.zeros(size, device=device)
        self.vfs = th.zeros(size+(reward_shape[1],), device=device)
        self.rewards = th.zeros(size+(reward_shape[1],), device=device)
        self.traces = th.zeros(size+(reward_shape[1],), device=device)
        self.advantages = th.zeros(size+(reward_shape[1],), device=device)

        self.num_workers = num_workers
        self.context_window = context_window


        self.steps_taken = th.zeros(
            num_workers,
            dtype=th.long, 
            device=device
        )
        self.gamma = gamma
        # Precompute gamm vector [1, gamma*lambda, (gamma*lambda)^2, ..., (gamma*lambda)^n]
        gamm_list = [1]
        gammlam_list = [1]
        for i in range(size[1]):
            gamm_list.append(math.pow(gamma, i+1))
            gammlam_list.append(math.pow(gamma*lam, i+1))
        # self.gamm = th.tensor(gamm_list, device=device)
        # self.gammlam = th.tensor(gammlam_list, device=device)

        # gammlam_trig
        gamm_trig = []
        gammlam_trig = []
        for i in reversed(range(size[1])):
            gamm_trig.append([0 for _ in range(size[1]-i-1)]+gamm_list[:i+1])
            gammlam_trig.append([0 for _ in range(size[1]-i-1)]+gammlam_list[:i+1])
        self.gamm_trig = th.tensor(gamm_trig, device=device) # LxL matrix
        self.gammlam_trig = th.tensor(gammlam_trig, device=device) # LxL matrix

        self.current_states = th.zeros((num_workers,)+ (size[1]+1,) + (states_shape[1],), device=device)
        self.current_actions = th.zeros((num_workers,)+ (size[1],) + (actions_shape[1],), device=device)
        self.current_log_probs = th.zeros((num_workers,)+ (size[1],), device=device)
        self.current_vfs = th.zeros((num_workers,)+ (size[1],) + (reward_shape[1],), device=device)
        self.current_rewards = th.zeros((num_workers,)+ (size[1],) + (reward_shape[1],), device=device)

        self.counter = 0
        self.batch_size = batch_size

    def step(self):
        self.steps_taken += 1

    def reset_worker(self, idx):
        length = self.steps_taken[idx].item()
        self.steps_taken[idx] = 0
        return length
    
    def reset(self):
        self.lengths.zero_()
        self.states.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.vfs.zero_()
        self.rewards.zero_()
        self.traces.zero_()
        self.advantages.zero_()
        self.current_states.zero_()
        self.current_actions.zero_()
        self.current_log_probs.zero_()
        self.current_vfs.zero_()
        self.current_rewards.zero_()

        self.steps_taken.zero_()
        self.counter = 0

    @property
    def full(self):
        return self.counter >= self.size[0] # B is supposed to be the final size of the batch. number of trayectories
    
    def fetch(self) -> th.Tensor:
        return (
            self.current_states[:, -self.context_window -1:], 
            self.current_actions[:, -self.context_window:], 
            self.current_rewards[:, -self.context_window:],
            self.current_vfs[:, -self.context_window:],
            F.relu(self.context_window - self.steps_taken)
        )
    
    def append(self, action, value_fn, log_prob, obs, reward, info):
        self.current_states[:, :-1] = self.current_states[:, 1:].clone()
        self.current_actions[:, :-1] = self.current_actions[:, 1:].clone()
        self.current_log_probs[:, :-1] = self.current_log_probs[:, 1:].clone()
        self.current_vfs[:, :-1] = self.current_vfs[:, 1:].clone()
        self.current_rewards[:, :-1] = self.current_rewards[:, 1:].clone()
        self.current_states[:, -1] = obs
        self.current_actions[:, -1] = action
        self.current_log_probs[:, -1] = log_prob
        self.current_vfs[:, -1] = value_fn
        self.current_rewards[:, -1] = th.tensor(reward, device=self.device).unsqueeze(-1)
    
    def add(self, idx, vf_last):
        jdx = self.counter
        length = int(self.steps_taken[idx].item())
        self.lengths[jdx] = self.steps_taken[idx]
        # Automatic change of batch size
        self.states[jdx] = self.current_states[idx]
        self.actions[jdx] = self.current_actions[idx]
        self.log_probs[jdx] = self.current_log_probs[idx]
        self.rewards[jdx] = self.current_rewards[idx]
        self.traces[jdx] = self.gammlam_trig @ self.rewards[jdx] 
        self.vfs[jdx] = self.current_vfs[idx]
        next_vfs = th.zeros_like(self.vfs[jdx])
        next_vfs[:-1] = self.vfs[jdx, 1:]
        next_vfs[-1] = vf_last
        deltas = self.rewards[jdx] + self.gamma * next_vfs - self.vfs[jdx]
        # Gae
        self.advantages[jdx] = self.gammlam_trig @ deltas
        if length < self.size[1]:
            self.advantages[jdx, :-length] = 0.0
            self.traces[jdx, :-length] = 0.0
        self.reset_worker(idx)
        self.counter += 1


    def set_initial_obs(self, obs, idx):
        self.current_states[idx].zero_()
        self.current_actions[idx].zero_()
        self.current_log_probs[idx].zero_()
        self.current_vfs[idx].zero_()
        self.current_rewards[idx].zero_()
        self.current_states[idx, -1] = obs

    def get_avg_reward_trace(self) -> float:
        if self.counter == 0:
            return 0.0

        returns = []
        max_len = self.size[1]
        for i in range(self.counter):
            length = int(self.lengths[i].item())
            if length == 0:
                continue

            # First valid timestep (after left-padding)
            start = max_len - length
            returns.append(self.traces[i, start, 0])

        if len(returns) == 0:
            return 0.0

        return th.stack(returns).mean().item()

    def to_static(self) -> DataLoader:
        chunked_states = []
        chunked_actions = []
        chunked_log_probs = []
        chunked_returns = []
        chunked_advantages = []
        chunked_rewards = []

        num_valid = self.counter
        W = self.context_window + 1
        max_len = self.size[1]

        valid_advantages = self.advantages[:num_valid]
        valid_vfs = self.vfs[:num_valid]
        returns = valid_advantages + valid_vfs
        
        for i in range(num_valid):
            length = int(self.lengths[i].item())
            offset = max_len - length
            if length == 0:
                continue
            """
            if length <= W:
                starts = [max_len - W]
            else:
                starts = list(range(offset, max_len, W))

                if starts[-1] + W > max_len:
                    starts[-1] = max_len - W
            """
            traj_states = self.states[i, offset:]
            traj_actions = self.actions[i, offset:]
            traj_log_probs = self.log_probs[i, offset:]
            traj_advs = self.advantages[i, offset:]
            traj_returns = returns[i, offset:]
            traj_rewards = self.rewards[i, offset:]
            for t in range(0, length+1 - W):
                s_slice = traj_states[t: W + t]
                a_slice = traj_actions[t: W + t]
                l_slice = traj_log_probs[t: W + t]
                ret_slice = traj_returns[t: W + t]
                adv_slice = traj_advs[t: W + t]
                rew_slice = traj_rewards[t: W + t]

                chunked_states.append(s_slice)
                chunked_actions.append(a_slice)
                chunked_log_probs.append(l_slice)
                chunked_returns.append(ret_slice)
                chunked_advantages.append(adv_slice)
                chunked_rewards.append(rew_slice)
            """                     
            for start in starts:
                end = start + W
                
                chunked_states.append(self.states[i, start:end])
                chunked_actions.append(self.actions[i, start:end])
                chunked_log_probs.append(self.log_probs[i, start:end])
                chunked_returns.append(self.advantages[i, start:end])
                chunked_advantages.append(advantages[i, start:end])
                chunked_rewards.append(self.rewards[i, start:end])
            """

        flat_states = th.stack(chunked_states)
        flat_actions = th.stack(chunked_actions)
        flat_log_probs = th.stack(chunked_log_probs)
        flat_returns = th.stack(chunked_returns)
        flat_advantages = th.stack(chunked_advantages)
        flat_rewards = th.stack(chunked_rewards)
        dataset = TensorDataset(
            flat_states, 
            flat_actions, 
            flat_log_probs, 
            flat_returns, 
            flat_advantages,
            flat_rewards
        )
    
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)