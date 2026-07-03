import numpy as np
 
def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
    dones = None,
) -> np.ndarray:
    """
    Externalized gae function for ease of handle
    """
    B, T = rewards.shape
 
    if values.shape != (B, T + 1):
        raise ValueError(
            f"values must have shape [B, T+1] = [{B}, {T + 1}], "
            f"got {values.shape}"
        )
 
    if dones is None:
        dones = np.zeros((B, T), dtype=rewards.dtype)
 
    if dones.shape != (B, T):
        raise ValueError(
            f"dones must have shape [B, T] = [{B}, {T}], got {dones.shape}"
        )
 
    not_done = 1.0 - dones.astype(rewards.dtype)
    advantages = np.zeros((B, T), dtype=rewards.dtype)
    last_gae   = np.zeros(B,      dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_val  = values[:, t + 1]                                  
        delta     = rewards[:, t] + gamma * not_done[:, t] * next_val - values[:, t] # [B]
        last_gae  = delta + gamma * lam * not_done[:, t] * last_gae 
        advantages[:, t] = last_gae
 
    return advantages
