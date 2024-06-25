import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

from rlkit.utils.buffer import TrajectoryBuffer

# Managing High-Task Complexity: Hierarchical Meta-RL via Skill Representational Learning 
# HiMeta (Hierarchical Meta-Reinforcement Learning)
class HiMeta(nn.Module):
    def __init__(self,
                 HLmodel: nn.Module,
                 ILmodel: nn.Module,
                 LLmodel: nn.Module,
                 traj_buffer: TrajectoryBuffer,
                 HL_lr: float = 1e-3, 
                 IL_lr: float = 5e-4, # VAE lr
                 actor_lr: float = 3e-4, # this is PPO policy agent
                 critic_lr: float = 3e-4, # this is PPO policy agent
                 ###params###
                 embed_dim: int = None,
                 tau: float = 0.95,
                 gamma: float = 0.99,
                 K_epochs: int = 5,
                 eps_clip: float = 0.1,
                 l2_reg: float = 1e-4,
                 batch_training: bool = False,
                 device=torch.device('cpu')):
        super(HiMeta, self).__init__()
        self.HLmodel = HLmodel
        self.ILmodel = ILmodel
        self.LLmodel = LLmodel
        self.traj_buffer = traj_buffer

        self.loss_fn = torch.nn.MSELoss()

        optim_params = [{'params': self.HLmodel.parameters(), 'lr': HL_lr},
                        {'params': self.ILmodel.parameters(), 'lr': IL_lr},
                        {'params': self.LLmodel.actor.parameters(), 'lr': actor_lr},
                        {'params': self.LLmodel.critic.parameters(), 'lr': critic_lr}]

        self.optimizers = torch.optim.AdamW(optim_params)

        self.tau = tau
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.l2_reg = l2_reg
        self.batch_training = batch_training
        self.embed_dim = embed_dim
        penalty = [-embed_dim + i for i in range(1, embed_dim + 1)]
        self.occupancy_penalty = torch.exp(torch.tensor(penalty))
        self.device = device

    def train(self) -> None:
        self.HLmodel.train()
        self.ILmodel.train()
        self.LLmodel.train()

    def eval(self) -> None:
        self.HLmodel.eval()
        self.ILmodel.eval()
        self.LLmodel.eval()

    def to_device(self, device=torch.device('cpu')):
        self.device = device
        self.HLmodel.change_device_info(device)
        self.ILmodel.change_device_info(device)
        self.LLmodel.change_device_info(device)
        self.to(device)
    
    def init_encoder_hidden_info(self):
        self.HLmodel.encoder.init_hidden_info()

    def get_grad_norm(self):
        model_names = ['HL_state_embed', 'HL_action_embed', 'HL_reward_embed', 'HL_encoder', 'HL_cat_layer', 'HL_Gumbel_layer',
                       'IL_pre_embed', 'IL_encoder', 'IL_mu_network', 'IL_logstd_network', 'IL_decoder', 'IL_post_embed',
                       'LL_actor', 'LL_critic']
        models = [self.HLmodel.state_embed, self.HLmodel.action_embed, self.HLmodel.reward_embed, self.HLmodel.encoder, self.HLmodel.cat_layer, self.HLmodel.Gumbel_layer,
                  self.ILmodel.pre_embed, self.ILmodel.encoder, self.ILmodel.mu_network, self.ILmodel.logstd_network, self.ILmodel.decoder, self.ILmodel.post_embed,
                  self.LLmodel.actor, self.LLmodel.critic]
        grad_norms = []

        for model in models:
            grads = []
            num_params = 0
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.data.view(-1))
                    num_params += param.numel()

            flat_grads = torch.cat(grads)
            grad_norm = torch.norm(flat_grads, 1).item()
            normalized_grad_norm = grad_norm / num_params
            grad_norms.append(normalized_grad_norm)
        
        grad_norm_dict = {os.path.join('grad_norm', name): datum for name, datum in zip(model_names, grad_norms)}

        return grad_norm_dict
    
    def get_weight_norm(self):
        model_names = ['HL_state_embed', 'HL_action_embed', 'HL_reward_embed', 'HL_encoder', 'HL_cat_layer', 'HL_Gumbel_layer',
                    'IL_pre_embed', 'IL_encoder', 'IL_mu_network', 'IL_logstd_network', 'IL_decoder', 'IL_post_embed',
                    'LL_actor_backbone', 'LL_actor_dist_net', 'LL_critic']
        models = [self.HLmodel.state_embed, self.HLmodel.action_embed, self.HLmodel.reward_embed, self.HLmodel.encoder, self.HLmodel.cat_layer, self.HLmodel.Gumbel_layer,
                self.ILmodel.pre_embed, self.ILmodel.encoder, self.ILmodel.mu_network, self.ILmodel.logstd_network, self.ILmodel.decoder, self.ILmodel.post_embed,
                self.LLmodel.actor.backbone, self.LLmodel.actor.dist_net, self.LLmodel.critic]
        weight_norms = []

        for model in models:
            weights = []
            num_params = 0
            for param in model.parameters():
                weights.append(param.data.view(-1))
                num_params += param.numel()

            flat_weights = torch.cat(weights)
            weight_norm = torch.norm(flat_weights, 1).item()
            normalized_weight_norm = weight_norm / num_params
            weight_norms.append(normalized_weight_norm)
        
        weight_norm_dict = {os.path.join('weight_norm', name): datum for name, datum in zip(model_names, weight_norms)}

        return weight_norm_dict

    def average_dict(self, dict_list):
        sums = {}
        counts = {}
        for d in dict_list:
            for key, value in d.items():
                if key in sums:
                    sums[key] += value
                    counts[key] += 1
                else:
                    sums[key] = value
                    counts[key] = 1
        averages = {key: sums[key] / counts[key] for key in sums}
        return averages
    
    def forward(self, input_tuple, deterministic=False):
        '''decision making framework using all hierarchy'''
        '''Input: tuple or tuple batch dim: 1 x (s, a, ns, r, m) or batch x (s, a, ...)'''

        '''HL-model first'''
        
        with torch.no_grad():
            # obs and its corresponding categorical inference
            states, y, _, _ = self.HLmodel(input_tuple) 

            '''IL-model'''        
            ego_states, z, _, _ = self.ILmodel(states.detach(), y.detach())

            '''LL-model'''
            states = torch.concatenate((ego_states, z), axis=-1)
            action, logprob = self.LLmodel.select_action(states.detach(), deterministic=deterministic)
        
        return action, logprob, (y, z)
        
    def embed(self, input_tuple):
        '''
        Used during the update (learn), since it does not need to 
        make an action but encodded obs or embedding itself.
        '''
        # HL
        states, y, loss_cat, loss_occ = self.HLmodel(input_tuple, is_batch=True)

        # IL
        ego_states, z, z_mu, z_std = self.ILmodel(states.detach(), y.detach())

        y_embedded_states = torch.concatenate((states, y), axis=-1) # for critic; s + y
        z_embedded_states = torch.concatenate((ego_states, z), axis=-1) # for actor; s_ego + z = S - s_other + z

        return states, y_embedded_states, z_embedded_states, (y, z, z_mu, z_std), (loss_cat, loss_occ)
    
    def learn(self, batch):
        from rlkit.utils.utils import estimate_advantages, estimate_episodic_value
        if self.batch_training:
            self.traj_buffer.push(batch)

        '''CALL DATA FOR ACTOR UPDATE'''
        states = torch.from_numpy(batch['states']).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)
        next_states = torch.from_numpy(batch['next_states']).to(self.device)
        rewards = torch.from_numpy(batch['rewards']).to(self.device)
        masks = torch.from_numpy(batch['masks']).to(self.device)
        logprobs = torch.from_numpy(batch['logprobs']).to(self.device)
        successes = torch.from_numpy(batch['successes']).to(self.device)

        mdp_tuple = (states, actions, next_states, rewards, masks)
        
        '''COMPUTE ADVANTAGES FOR POLICY UPDATE'''
        with torch.no_grad():
            _, y_embedded_states, _, _, _ = self.embed(mdp_tuple) # for LL actor
            values = self.LLmodel.critic(y_embedded_states)

        '''GET THE ACTOR ADVANTAGE ESTIMATION'''
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)
        episodic_reward = estimate_episodic_value(rewards, masks, 1.0, self.device)

        grad_norm_list = []
        '''Update the parameters'''
        for _ in range(self.K_epochs):
            if self.batch_training:
                '''CALL TRAJ FROM THE BUFFER'''
                traj_batch = self.traj_buffer.sample(40)

                traj_states = torch.from_numpy(traj_batch['states']).to(self.device)
                traj_actions = torch.from_numpy(traj_batch['actions']).to(self.device)
                traj_next_states = torch.from_numpy(traj_batch['next_states']).to(self.device)
                traj_rewards = torch.from_numpy(traj_batch['rewards']).to(self.device)
                traj_masks = torch.from_numpy(traj_batch['masks']).to(self.device)
            
                traj_mdp_tuple = traj_states, traj_actions, traj_next_states, traj_rewards, traj_masks
                
                '''EMBEDD OVER HIMETA TO GET Y AND Z'''
                _, traj_y_embedded_states, _, (traj_y, traj_z, traj_z_mu, traj_z_std), _ = self.embed(traj_mdp_tuple) # for HL, IL, and LL critic
                _, _, z_embedded_states, _, _ = self.embed(mdp_tuple) # for LL actor

                '''COMPUTE THE CRITIC LOSS THAT TRAINS THE HIGH-LEVEL MODEL AND THE CRITIC ITSELF'''
                r_pred = self.LLmodel.critic(traj_y_embedded_states) # y embedding b/c it is sub-task info while z is action-task info; this is not action-value fn
                _, traj_returns = estimate_advantages(traj_rewards, traj_masks, r_pred, self.gamma, self.tau, self.device)
                value_loss = self.loss_fn(r_pred, traj_returns)

                '''COMPUTE THE VAE (INTERMEDIATE LEVEL) LOSS'''
                (decoder_loss, state_pred_loss, kl_loss) = self.ILmodel.decode(traj_states, traj_next_states, traj_z, traj_z_mu, traj_z_std)
            else:
                _, y_embedded_states, z_embedded_states, (y, z, z_mu, z_std), (loss_cat, loss_occ) = self.embed(mdp_tuple) # for LL actor

                '''COMPUTE THE CRITIC LOSS THAT TRAINS THE HIGH-LEVEL MODEL AND THE CRITIC ITSELF'''
                r_pred = self.LLmodel.critic(y_embedded_states) # y embedding b/c it is sub-task info while z is action-task info; this is not action-value fn
                _, traj_returns = estimate_advantages(rewards, masks, r_pred, self.gamma, self.tau, self.device)
                value_loss = self.loss_fn(r_pred, returns)

                '''COMPUTE THE VAE (INTERMEDIATE LEVEL) LOSS'''
                (decoder_loss, state_pred_loss, kl_loss) = self.ILmodel.decode(states, next_states, z, z_mu, z_std)

            '''COMPUTE THE ACTOR LOSS'''
            dist = self.LLmodel.actor(z_embedded_states.detach())

            new_logprobs = dist.log_prob(actions)            
            dist_entropy = dist.entropy()
    
            ratios = torch.exp(new_logprobs - logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            actor_loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * value_loss - 0.001 * dist_entropy)            
            #print(new_logprobs.shape, logprobs.shape, ratios.shape, surr1.shape, surr2.shape, advantages.shape, dist_entropy.shape, value_loss.shape, decoder_loss.shape)

            '''LOSS IS ADDED; THIS IS FINE SINCE THEY ARE CONNECTED TO THE DIFFERENT PARAMETERS''' 
            loss = actor_loss + decoder_loss + loss_cat + loss_occ

            '''UPDATE THE ALL MODELS'''
            self.optimizers.zero_grad()
            loss.backward()
            grad_norm_list.append(self.get_grad_norm())
            self.optimizers.step()

        training_output = {
            'loss/critic_loss': value_loss.item(),
            'loss/actor_loss': actor_loss.item(),
            'loss/decoder_loss': decoder_loss.item(),
            'loss/decoder_pred_loss': state_pred_loss.item(),
            'loss/decoder_kl_loss': kl_loss.item(),
            'loss/occupancy_loss': loss_occ.item(),
            'loss/cat_ent_loss': loss_cat.item(),
            'train/episodic_reward': episodic_reward.item(),
            'train/success': successes.mean().item()
        }

        grad_norm = self.average_dict(grad_norm_list)
        param_norm = self.get_weight_norm()

        result = {**training_output, **grad_norm, **param_norm}
        
        return result
    
    def save_model(self, logdir, epoch, is_best=False):
        self.actor, self.critic = self.LLmodel.actor.cpu(), self.LLmodel.critic.cpu()
        self.ILmodel = self.ILmodel.cpu()
        self.HLmodel = self.HLmodel.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump((self.actor, self.critic, self.ILmodel, self.HLmodel), open(path, 'wb'))

        self.actor, self.critic = self.actor.to(self.device), self.critic.to(self.device)
        self.ILmodel = self.ILmodel.to(self.device)
        self.HLmodel = self.HLmodel.to(self.device)