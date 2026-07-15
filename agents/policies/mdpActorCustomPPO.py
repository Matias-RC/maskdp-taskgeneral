from agents.maskdp.rewards_joint_PE import JointDPAgent, MaskDPJointPE
from utils.utils import get_1d_sincos_pos_embed_from_grid, to_torch
from agents.modules.attention import Block
import torch.nn as nn
import torch as th
import random


class ExtraDecoderForValueHead(nn.Module):
    def __init__(self, model: MaskDPJointPE, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.mask_token = nn.Parameter(model.mask_token.data.clone())
        self.mod_embed_s = nn.Parameter(model.mod_embed_s.clone())
        self.mod_embed_a = nn.Parameter(model.mod_embed_a.clone())
        self.mod_embed_r = nn.Parameter(model.mod_embed_r.clone())

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
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, 1),
        )
        self.value_head.load_state_dict(model.reward_head.state_dict())
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_time_steps)
        self.decoder_pos_embed = th.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        self.optimizer = th.optim.Adam(self.parameters(), lr=config.lr)
        
    def decodeValue(self, x, ids_restore):
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


        curr_len = x.shape[1]
        attn_mask = self.attn_mask[:, :, :curr_len, :curr_len]

        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)

        return self.value_head(x[:, -1])





class MDPWrapperPPO(JointDPAgent):
    """
    Reuses the reward prediction heads as proxy for the value function
    """
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
        pretrained=False,
        vf_head_type="transformer_decoder"
    ):
        super().__init__(
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        mask_ratio,
        transformer_cfg,
        )
        if pretrained:
            assert transformer_cfg.weights_path is not None
            pretrained = th.load(transformer_cfg.weights_path, map_location=transformer_cfg.device)
            self.model.load_state_dict(pretrained["model"], stric=True)
        self.vf_hed_type = vf_head_type
        self.value_head = ExtraDecoderForValueHead(self.model, transformer_cfg)
        self.zero_pad_prob = transformer_cfg.zero_pad_prob

    
    def act(self, states, actions, rewards, values): # sarv: Satates, Actions Rewards, Values
        actions = th.cat([actions, th.zeros_like(actions[:, -1:])], dim=1)
        rewards = th.cat([values, th.zeros_like(rewards[:, -1:])], dim=1)
        x, mask, ids_restore, num_mods = self.model.forward_encoder(states, actions, rewards, mask_ratio=0.0, finetune_input=True, do_last_n=2)
        _, latents, _ = self.model.forward_decoder(x, ids_restore, num_mods)
        pi = latents[:, -1]
        vf = self.value_head.decodeValue(x, ids_restore)
        return pi, vf
    
    def _prepare_inputs(self, states, actions, rewards):
            B, T, _ = states.shape
            
            actions_padded = th.cat([actions, th.zeros_like(actions[:, -1:])], dim=1)
            rewards_padded = th.cat([rewards, th.zeros_like(rewards[:, -1:])], dim=1)
            states_padded = states.clone()

            if random.random() < self.zero_pad_prob and T > 2:
                # Select a random number of trailing history steps to zero-out
                k = random.randint(1, T - 2)
                
                # Zero out from index (T - k - 1) up to the second-to-last index
                # We keep the very last step intact because it represents the active timestep
                states_padded[:, :k] = 0.0
                actions_padded[:, :k] = 0.0
                rewards_padded[:, :k] = 0.0

            return states_padded, actions_padded, rewards_padded
    
    def update_value_head(self, replay_iter, global_step):
            metrics = dict()
            batch = next(replay_iter)
            
            states, actions, latents, returns, advantages, rewards = to_torch(batch, self.device)
            states, actions, rewards = self._prepare_inputs(states, actions, rewards)
            actions_padded = th.cat([actions, th.zeros_like(actions[:, -1:])], dim=1)
            rewards_padded = th.cat([rewards, th.zeros_like(rewards[:, -1:])], dim=1)

            # Forward pass using actual rewards  from the buffer
            x, mask, ids_restore, num_mods = self.model.forward_encoder(
                states, actions_padded, rewards_padded, 
                mask_ratio=0.0, finetune_input=True, do_last_n=2
            )

            # Predict values
            predicted_values = self.value_head.decodeValue(x, ids_restore)  # Shape: [Batch, 1]
            
            # Match returns shape (typically [Batch])
            predicted_values = predicted_values.squeeze(-1)

            # Compute Loss (MSE against returns target)
            value_loss = nn.functional.mse_loss(predicted_values, returns)

            # Gradient Step
            self.value_head.optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
            self.value_head.optimizer.step()

            # Log metrics
            metrics["value_loss"] = value_loss.item()
            metrics["v_mean"] = predicted_values.mean().item()
            
            return metrics