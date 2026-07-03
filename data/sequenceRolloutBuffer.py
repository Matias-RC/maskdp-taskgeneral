import torch
import numpy as np

class SequenceRB:
    def __init__(self, size, s_shape, a_shape, timeout_limit, device="cpu"):
        self.size = size
        self.timeout_limit = timeout_limit
        self.device = device
        
        # Buffers: Note that states and values need timeout_limit + 1 
        # to account for the final bootstrap/terminal state for GAE
        self.states_buffer = torch.zeros((size, timeout_limit + 1, s_shape), dtype=torch.float32, device=device)
        self.actions_buffer = torch.zeros((size, timeout_limit, a_shape), dtype=torch.float32, device=device)
        self.rewards_buffer = torch.zeros((size, timeout_limit, 1), dtype=torch.float32, device=device)
        self.values_buffer = torch.zeros((size, timeout_limit + 1, 1), dtype=torch.float32, device=device)
        self.dones_buffer = torch.zeros((size, timeout_limit, 1), dtype=torch.float32, device=device)
        
        # Track actual episode lengths (since episodes can terminate early)
        self.lens_buffer = torch.zeros((size,), dtype=torch.long, device=device)
        
        self.ptr = 0
        self.count = 0

    def upload(self, s_seq, a_seq, r_seq, vf_seq, last_val):
        """
        Uploads a complete episode sequence collected from rollout.
        Expected shapes (assuming single environment batch dimension B=1):
            s_seq:  [1, T + 1, s_shape]
            a_seq:  [1, T, a_shape]
            r_seq:  [1, T, 1]
            vf_seq: [1, T, 1]
            last_val: [1, 1, 1] (The value prediction V(s_{T+1}))
        """
        # Squeeze out the environment batch dimension (B=1) from rollout tracking
        s = s_seq.squeeze(0)
        a = a_seq.squeeze(0)
        r = r_seq.squeeze(0)
        vf = vf_seq.squeeze(0)
        
        T = a.shape[0]

        if T > self.timeout_limit:
            raise ValueError(f"Episode length {T} exceeds timeout limit {self.timeout_limit}")

        # Construct full value trajectory including the trailing bootstrap value
        full_vf = torch.cat([vf, last_val], dim=0)

        # Insert trajectories into storage arrays
        self.states_buffer[self.ptr, :T+1] = s
        self.actions_buffer[self.ptr, :T] = a
        self.rewards_buffer[self.ptr, :T] = r
        self.values_buffer[self.ptr, :T+1] = full_vf
        
        # Mark the last environment interaction step as done
        self.dones_buffer[self.ptr, :T] = 0.0
        self.dones_buffer[self.ptr, T-1] = 1.0 
        
        self.lens_buffer[self.ptr] = T

        # Cycle writing index pointer
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def reset(self):
        self.states_buffer.zero_()
        self.actions_buffer.zero_()
        self.rewards_buffer.zero_()
        self.values_buffer.zero_()
        self.dones_buffer.zero_()
        self.lens_buffer.zero_()
        self.ptr = 0
        self.count = 0

    @property
    def full(self):
        return self.count >= self.size

    def get_all_data(self):
        """Returns all completed episode slots for calculating GAE."""
        idx = slice(0, self.count)
        return (
            self.states_buffer[idx],
            self.actions_buffer[idx],
            self.rewards_buffer[idx],
            self.values_buffer[idx],
            self.dones_buffer[idx],
            self.lens_buffer[idx],
        )