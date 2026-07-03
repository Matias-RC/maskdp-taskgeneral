import numpy as np
import torch
from ppo.gae import gae

class MaskDPTrainer:
    def __init__(
        self,
        agent,
        env,
        buffer,
        device,
        cfg,
    ):
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.cfg = cfg
        self.device = device
        self.s_seq = None
        self.a_seq = None
        self.r_seq = None
        self.vf_seq = None

        self.max_history = self.agent.backbone.traj_length
        self.environment_healthy = False
    
    def rollout_collection(self):
        assert self.s_seq is not None and self.a_seq is not None and self.r_seq is not None

        self.buffer.reset()

        # Lists to store episodic averages across the entire rollout batch phase
        rollout_episode_rewards = []
        rollout_episode_err_norms = []

        # Temporary lists tracking the *current* ongoing episode step variables
        current_ep_rewards = []
        current_ep_err_norms = []

        while not self.buffer.full:
            s_ctx = self.s_seq[:, -self.max_history:]
            a_ctx = self.a_seq[:, -self.max_history+1:]
            r_ctx = self.r_seq[:, -self.max_history+1:]

            pi, vf = self.agent.act(s_ctx, a_ctx, r_ctx)
            action_np = pi.detach().cpu().numpy()[0]
            
            time_step = self.env.step(action_np)
            next_obs = time_step.observation
            reward = time_step.reward
            done = time_step.last()
            
            if isinstance(reward, np.ndarray) and reward.size == 0:
                reward = 0.0
            elif reward is None:
                reward = 0.0
            else:
                if not self.environment_healthy:
                    print("Environment is healthy...")
                    self.environment_healthy = True

            # =================================================================
            # TELEMETRY EXTRACTION: Safely grab err_norm (distance from goal)
            # =================================================================
            err_norm = 0.0
            # Case A: Environment returns extra metrics in a standard .info dictionary
            if hasattr(time_step, 'info') and isinstance(time_step.info, dict):
                err_norm = time_step.info.get('err_norm', time_step.info.get('distance_from_goal', 0.0))
            # Case B: Observation space is structured as a dictionary containing the goal error
            elif isinstance(next_obs, dict):
                err_norm = next_obs.get('err_norm', next_obs.get('distance_from_goal', 0.0))
            # Case C: If your observation is a flat array and err_norm is located at a specific index
            # elif isinstance(next_obs, np.ndarray):
            #     err_norm = next_obs[TARGET_INDEX]

            current_ep_rewards.append(float(reward))
            current_ep_err_norms.append(float(err_norm))
            # =================================================================
                    
            action_t = pi.view(1, 1, -1)
            value_t = vf.view(1, 1, -1)
            reward_t = torch.as_tensor([reward], device=self.device, dtype=torch.float32).view(1, 1, 1)
            
            self.a_seq = torch.cat([self.a_seq, action_t], dim=1)
            self.r_seq = torch.cat([self.r_seq, reward_t], dim=1)
            self.vf_seq = torch.cat([self.vf_seq, value_t], dim=1)

            if done:
                next_obs_t = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32).view(1, 1, -1) if not isinstance(next_obs, dict) else torch.as_tensor(next_obs['observation'], device=self.device, dtype=torch.float32).view(1, 1, -1)
                self.s_seq = torch.cat([self.s_seq, next_obs_t], dim=1)
                
                s_ctx_final = self.s_seq[:, -self.max_history:]
                a_ctx_final = self.a_seq[:, -self.max_history+1:]
                r_ctx_final = self.r_seq[:, -self.max_history+1:]
                
                with torch.no_grad():
                    _, last_val = self.agent.act(s_ctx_final, a_ctx_final, r_ctx_final)
                
                self.buffer.upload(self.s_seq, self.a_seq, self.r_seq, self.vf_seq, last_val)
                
                # Append finalized episodic averages before clearing workspace lists
                if len(current_ep_rewards) > 0:
                    rollout_episode_rewards.append(np.mean(current_ep_rewards))
                    rollout_episode_err_norms.append(np.mean(current_ep_err_norms))
                
                current_ep_rewards = []
                current_ep_err_norms = []
                
                self.setup_reset()
                continue

            next_obs_t = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32).view(1, 1, -1) if not isinstance(next_obs, dict) else torch.as_tensor(next_obs['observation'], device=self.device, dtype=torch.float32).view(1, 1, -1)
            self.s_seq = torch.cat([self.s_seq, next_obs_t], dim=1)

        total_steps_collected = len(rollout_episode_rewards) * self.max_history
        metrics = {
            "mean_reward": float(np.mean(rollout_episode_rewards)) if rollout_episode_rewards else 0.0,
            "mean_err_norm": float(np.mean(rollout_episode_err_norms)) if rollout_episode_err_norms else 0.0,
            "num_steps": total_steps_collected
        }
        return True, metrics

    def setup_reset(self):
        time_step = self.env.reset()
        obs = time_step.observation if hasattr(time_step, 'observation') else time_step[0]
        if isinstance(obs, dict):
            obs = obs['observation']
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).view(1, 1, -1)
        
        self.s_seq = obs
        self.a_seq = torch.zeros((1, 0, self.env.action_spec().shape[0]), device=self.device)
        self.r_seq = torch.zeros((1, 0, 1), device=self.device)
        self.vf_seq = torch.zeros((1, 0, 1), device=self.device)
    
    def finetuning(self, epochs):
        states, actions, rewards, values, dones, lens = self.buffer.get_all_data()
        
        rewards_np = rewards.squeeze(-1).cpu().numpy()  
        values_np = values.squeeze(-1).cpu().numpy()    
        dones_np = dones.squeeze(-1).cpu().numpy()      

        advantages_np = gae(rewards_np, values_np, gamma=self.cfg.gamma, lam=self.cfg.lam, dones=dones_np)
        advantages = torch.as_tensor(advantages_np, device=self.device)

        sub_states, sub_actions, sub_rewards, sub_values, sub_advantages = [], [], [], [], []
        num_trajectories = states.shape[0]

        for i in range(num_trajectories):
            traj_len = int(lens[i].item() if hasattr(lens[i], 'item') else lens[i])
            if traj_len < self.max_history:
                continue
            
            starts = list(range(0, traj_len - self.max_history, self.max_history))
            starts.append(traj_len - self.max_history)
            starts = sorted(list(set(starts)))
            for start in starts:
                end = start + self.max_history
                
                sub_states.append(states[i, start:end])
                sub_actions.append(actions[i, start:end])
                sub_rewards.append(rewards[i, start:end])
                sub_values.append(values[i, start:end].squeeze(-1) if values.dim() == 3 else values[i, start:end])  
                sub_advantages.append(advantages[i, start:end])
        
        if len(sub_states) == 0:
            raise RuntimeError("No valid trajectory sequences captured for optimization window chunking.")
            
        sub_states = torch.stack(sub_states, dim=0)
        sub_actions = torch.stack(sub_actions, dim=0)
        sub_rewards = torch.stack(sub_rewards, dim=0)
        sub_values = torch.stack(sub_values, dim=0)
        sub_advantages = torch.stack(sub_advantages, dim=0)
        print(sub_advantages.shape)
        num_samples = sub_states.shape[0]
        batch_size = getattr(self.cfg, 'batch_size', 64)
        
        # Precompute target metrics
        sub_old_log_probs = []
        sub_old_values = []
        
        self.agent.eval()
        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                b_states = sub_states[start_idx:end_idx]
                b_actions = sub_actions[start_idx:end_idx, :-1]
                b_rewards = sub_rewards[start_idx:end_idx, :-1]

                action_mean, v_preds = self.agent._forward_transformer(b_states, b_actions, b_rewards)

                action_mean = action_mean[:, :-1] # Slice it because it has a longer length and b_actions
                v_preds = v_preds[:, :-1].squeeze(-1)

                action_std = self.agent.actor_logstd.exp().expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = dist.log_prob(b_actions).sum(dim=-1)
                
                sub_old_log_probs.append(log_probs)
                sub_old_values.append(v_preds)
                
        sub_old_log_probs = torch.cat(sub_old_log_probs, dim=0)
        sub_old_values = torch.cat(sub_old_values, dim=0)

        target_kl = getattr(self.cfg, 'target_kl', 0.015)
        data = []
        
        for j in range(epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            epoch_kls = []
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = permutation[start_idx : start_idx + batch_size]
                
                b_states = sub_states[batch_indices]
                b_actions = sub_actions[batch_indices, :-1]
                b_rewards = sub_rewards[batch_indices, :-1]
                b_advantages = sub_advantages[batch_indices, :-1]

                b_old_log_probs = sub_old_log_probs[batch_indices]
                b_old_values = sub_old_values[batch_indices]
                metrics = self.agent.predict(
                    b_states, b_actions, b_rewards, b_advantages, b_old_log_probs, b_old_values
                )
                data.append(metrics)
                epoch_kls.append(metrics["approx_kl"])
                
            if np.mean(epoch_kls) > target_kl:
                print(f"Early stopping at epoch {j+1}/{epochs}. Mean KL: {np.mean(epoch_kls):.4f} > {target_kl}")
                break
                
        return data

    def train(self, iterations, epochs, callback=None):
        for i in range(iterations): 
            self.setup_reset()
            
            # Rollout tracking captures both status and runtime metrics
            continue_training, env_metrics = self.rollout_collection()

            if not continue_training:
                break

            # Finetuning execution
            data = self.finetuning(epochs)

            # Pass both sequence gradient metrics and physical environment metrics forward
            if callback is not None:
                callback(iteration=i + 1, losses_data=data, env_metrics=env_metrics, agent=self.agent)

        print("Training complete!")