from torch.utils.data import TensorDataset, DataLoader
from data.online_rollout_buffer import CustomPPO_RB
import torch.nn.functional as F
from collections import deque
import torch as th
import numpy as np
import math

class CustomRB_MaxStepBased(CustomPPO_RB):
    def __init__(self, max_steps=2048, **kwargs):
        super().__init__(**kwargs)
        self.lengths = deque()
        self.states = deque()
        self.actions = deque()
        self.log_probs = deque()
        self.vfs = deque()
        self.rewards = deque()
        self.traces = deque()
        self.advantages = deque()
        self.max_steps = max_steps
        self.outer_loop_steps_counter = 0

    @property
    def full(self):
        return self.outer_loop_steps_counter >= self.max_steps

    def step(self):
        self.outer_loop_steps_counter += 1
        super().setp()

    def add(self, idx, vf_last):
        jdx = self.counter
        length = int(self.steps_taken[idx].item())
        self.lengths.append(self.steps_taken[idx])
        # Automatic change of batch size
        self.states.append(self.current_states[idx])
        self.actions.append(self.current_actions[idx])
        self.log_probs.append(self.current_log_probs[idx])
        self.rewards.append(self.current_rewards[idx])
        
        self.vfs.append(self.current_vfs[idx])
        next_vfs = th.zeros_like(self.current_vfs[idx])
        next_vfs[:-1] = self.current_vfs[idx, 1:]
        next_vfs[-1] = vf_last
        deltas = self.current_rewards[idx] + self.gamma * next_vfs - self.current_vfs[idx]
        # Gae
        trcs = self.gammlam_trig @ self.current_rewards[idx]
        advtgs = self.gammlam_trig @ deltas

        if length < self.size[1]:
            advtgs[:-length] = 0.0
            trcs[:-length] = 0.0
        self.advantages.append(advtgs)
        self.traces.append(trcs)

        self.reset_worker(idx)
        self.counter += 1

    def to_static(self) -> DataLoader:
        # We have some leftover trajectories that didn't get to finish, thus we incorporate them before creating the torch dataloader object
        self.current_actions = self.current_actions[:, :-1].clone()
        vf_last = self.current_vfs[:, -1].clone()
        self.current_vfs = self.current_vfs[:, :-1].clone()
        self.current_states = ...
        for i in range(self.num_workers):
            self.add(idx=i, vf_last=vf_last[i])
                

