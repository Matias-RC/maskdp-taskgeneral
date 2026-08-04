from agents.maskdp.rewards_joint_PE import JointDPAgent, MaskDPJointPE
from utils.utils import get_1d_sincos_pos_embed_from_grid, to_torch
from agents.modules.attention import Block
import torch.nn as nn
import torch as th
import random
import math

class ExtraDecoderForValueHead(nn.Module):
    def __init__(self, model: MaskDPJointPE, config, lr):
        super().__init__()
        self.n_embd = config.n_embd
        self.mask_token = nn.Parameter(model.mask_token.data.clone())
        self.mod_embed_s = nn.Parameter(model.mod_embed_s.data.clone())
        self.mod_embed_a = nn.Parameter(model.mod_embed_a.data.clone())
        self.mod_embed_r = nn.Parameter(model.mod_embed_r.data.clone())

        self.decoder_state_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_reward_embed = nn.Linear(self.n_embd, self.n_embd)

        self.decoder_state_embed.load_state_dict(model.decoder_state_embed.state_dict())
        self.decoder_action_embed.load_state_dict(model.decoder_action_embed.state_dict())
        self.decoder_reward_embed.load_state_dict(model.decoder_reward_embed.state_dict())

        self.decoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_dec_layer)]
        )
        self.decoder_blocks.load_state_dict(model.decoder_blocks.state_dict())
        
        # Initialize Value Head to output a scalar state value V(s)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, 1),
        )
        # Use existing reward head weights for a stable initialization
        self.value_head.load_state_dict(model.reward_head.state_dict())
        
        # Pull identical positional embeddings and attention masks from reference model
        self.register_buffer("decoder_pos_embed", model.decoder_pos_embed.clone())
        self.register_buffer("attn_mask", model.attn_mask.clone())

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)
        
    def decodeValue(self, x, ids_restore, attn_mask=None):
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = th.cat([x, mask_tokens], dim=1)
        x = th.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        T = x.shape[1] // 3

        s = self.decoder_state_embed(x[:, 0::3]) + self.mod_embed_s + self.decoder_pos_embed[:, :T, :]
        a = self.decoder_action_embed(x[:, 1::3]) + self.mod_embed_a + self.decoder_pos_embed[:, :T, :]
        r = self.decoder_reward_embed(x[:, 2::3]) + self.mod_embed_r + self.decoder_pos_embed[:, :T, :]
        x = th.stack([s, a, r], dim=2).reshape_as(x)
        if attn_mask is None:
            curr_len = x.shape[1]
            attn_mask = self.attn_mask[:, :, :curr_len, :curr_len]

        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)

        # Extract the representation of the most recent state token
        states_decoded = x[:, 0::3]
        last_state_repr = states_decoded[:, -1] # Shape: [B, n_embd]

        return self.value_head(last_state_repr)


class MDPWrapperPPO(JointDPAgent):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        mask_ratio,
        transformer_cfg,
        name,
        pretrained=False,
        vf_head_type="transformer_decoder",
        std_lr=1e-2,
    ):
        super().__init__(
            (obs_shape[1],),
            (action_shape[1],),
            device,
            lr,
            batch_size,
            use_tb,
            mask_ratio,
            transformer_cfg,
            name,
        )
        if pretrained:
            assert transformer_cfg.weights_path is not None
            pretrained_dict = th.load(transformer_cfg.weights_path, map_location=device)
            self.model.load_state_dict(pretrained_dict["model"], strict=True)
            
        self.vf_head_type = vf_head_type
        self.value_head = ExtraDecoderForValueHead(self.model, transformer_cfg, lr).to(device)
        self.zero_pad_prob = transformer_cfg.zero_pad_prob

        # Trainable parameter for policy standard deviation
        action_dim = action_shape[1]
        self.log_std = nn.Parameter(th.zeros(action_dim, device=device)+ math.log(transformer_cfg.initial_std))
        self.std_lr = std_lr
        # Add log_std to the policy optimizer created in super().__init__()
        if hasattr(self, "opt") and self.opt is not None:
            self.opt.add_param_group({"params": [self.log_std], "lr": self.std_lr})

    @property
    def std(self):
        return self.log_std.clamp(-20, 2).exp()

    def _prepare_inputs(self, states: th.Tensor, actions: th.Tensor, rewards: th.Tensor, training=True):
        B, T, _ = states.shape
        if not training:
            # Pad actions and rewards to match sequence length
            actions_padded = th.cat([actions, th.zeros_like(actions[:, -1:])], dim=1)
            rewards_padded = th.cat([rewards, th.zeros_like(rewards[:, -1:])], dim=1)
            states_padded = states.clone()
            return states_padded, actions_padded, rewards_padded
        
        if random.random() < self.zero_pad_prob and T > 2:
            k = random.randint(1, T - 2)
            pad_len = T - k 
            #states_pad = th.zeros(B, pad_len, states.shape[-1], device=states.device)
            #actions_pad = th.zeros(B, pad_len, actions.shape[-1], device=actions.device)
            #rewards_pad = th.zeros(B, pad_len, rewards.shape[-1], device=rewards.device)
            #states = th.cat((states_pad, states[:, :k]), dim=1)
            #actions = th.cat((actions_pad, actions[:, :k]), dim=1)
            #rewards = th.cat((rewards_pad, rewards[:, :k]), dim=1)
            return states[:, :k].clone(), actions[:, :k].clone(), rewards[:, :k].clone()

        return states, actions, rewards
    
    def act(self, states, actions, rewards, values=None, pad_lengths=None): 
        with th.no_grad():
            states_p, actions_p, rewards_p = self._prepare_inputs(states, actions, rewards, training=False)
            
            # Run encoder
            x, mask, ids_restore, num_mods, attn_mask = self.model.forward_encoder(
                states_p, actions_p, rewards_p, mask_ratio=0.0, finetune_input=True, do_last_n=2, pad_lengths=pad_lengths
            )
            
            # Policy output: Forward original decoder to predict action mean
            _, a_pred, _ = self.model.forward_decoder(x, ids_restore, num_mods, pad_lengths=pad_lengths)

            pi_mean = a_pred[:, -1]
            
            # Construct policy action distribution
            dist = th.distributions.Normal(pi_mean, self.std)
            
            # Value output: Forward custom value decoder
            vf = self.value_head.decodeValue(x, ids_restore)
            
        return dist, vf

    def update_value_head(self, batch, global_step):
        metrics = dict()
        
        states, actions, latents, returns, advantages, rewards = to_torch(batch, self.device)

        states_p, actions_p, rewards_p = self._prepare_inputs(states, actions, rewards, training=True)

        # Freeze the main encoder to optimize ONLY the value-specific decoder parameters
        with th.no_grad():
            x, mask, ids_restore, num_mods, _ = self.model.forward_encoder(
                states_p, actions_p, rewards_p, 
                mask_ratio=0.0, finetune_input=True, do_last_n=2
            )

        # Predict values V(s) from the target representations
        predicted_values = self.value_head.decodeValue(x, ids_restore).squeeze(-1)
        target_returns = returns[:, -1]
        # Mean Squared Error Loss against returns
        value_loss = nn.functional.mse_loss(predicted_values, target_returns)

        # Step value function optimizer
        self.value_head.optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
        self.value_head.optimizer.step()

        metrics["value_loss"] = value_loss.item()
        metrics["v_mean"] = predicted_values.mean().item()
        
        return metrics

    def update_policy(self, batch, global_step):
        metrics = dict()

        states, actions, old_log_probs, returns, advantages, rewards = to_torch(batch, self.device)
        states_p, actions_p, rewards_p = self._prepare_inputs(states, actions, rewards, training=True)
        x, mask, ids_restore, num_mods, _ = self.model.forward_encoder(
            states_p, actions_p, rewards_p, 
            mask_ratio=0.0, pad_lengths=None, 
            finetune_input=True, do_last_n=2
        )

        s, a, r = self.model.forward_decoder(x, ids_restore, 3)

        pi_mean = a[:, -1]
        target_actions = actions[:, -1]
        target_advantages = advantages[:, -1]

        target_advantages = (target_advantages - target_advantages.mean()) / (target_advantages.std() + 1e-8)

        # Use the trainable standard deviation
        current_std = self.std
        dist = th.distributions.Normal(pi_mean, current_std)
        log_probs = dist.log_prob(target_actions).sum(dim=-1)

        log_ratio = log_probs - old_log_probs[:, -1]

        ratio = th.exp(log_ratio)
        
        clip_coef = self.config.clip_coef
        
        surr1 = ratio * target_advantages
        surr2 = th.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * target_advantages
        policy_loss = -th.min(surr1, surr2).mean()

        entropy = dist.entropy().mean()
        ent_coef = getattr(self.config, "ent_coef", 0.0)
        
        loss = policy_loss - ent_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.opt.step()

        with th.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        metrics["policy_loss"] = policy_loss.item()
        metrics["entropy"] = entropy.item()
        metrics["approx_kl"] = approx_kl
        metrics["clipfrac"] = clipfrac
        metrics["pi_mean"] = pi_mean.mean().item()
        metrics["std"] = current_std.mean().item()

        return metrics