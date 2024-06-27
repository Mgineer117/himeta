import numpy as np
import os
import time
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from rlkit.utils.zfilter import ZFilter
from typing import Dict, List, Union, Tuple, Optional

# Managing High-Task Complexity: Hierarchical Meta-RL via Skill Representational Learning 
# HiMeta (Hierarchical Meta-Reinforcement Learning)
class HiMeta(nn.Module):
    def __init__(self,
                 HLmodel: nn.Module,
                 ILmodel: nn.Module,
                 LLmodel: nn.Module,
                 HL_lr: float = 1e-3, 
                 IL_lr: float = 5e-4, # VAE lr
                 actor_lr: float = 3e-4, # this is PPO policy agent
                 critic_lr: float = 3e-4, # this is PPO policy agent
                 ###params###
                 tau: float = 0.95,
                 gamma: float = 0.99,
                 K_epochs: int = 5,
                 eps_clip: float = 0.1,
                 entropy_scaler:float = 0.001,
                 l2_reg: float = 1e-4,
                 ###etc###
                 state_scaler: ZFilter = None,
                 reward_scaler: ZFilter = None,
                 device=torch.device('cpu')):
        super(HiMeta, self).__init__()
        self.HLmodel = HLmodel
        self.ILmodel = ILmodel
        self.LLmodel = LLmodel

        self.loss_fn = torch.nn.MSELoss()

        optim_params = [{'params': self.HLmodel.parameters(), 'lr': HL_lr},
                        {'params': self.ILmodel.parameters(), 'lr': IL_lr},
                        {'params': self.LLmodel.actor.parameters(), 'lr': actor_lr},
                        {'params': self.LLmodel.critic.parameters(), 'lr': critic_lr}]

        self.optimizers = torch.optim.Adam(optim_params)

        self.tau = tau
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.entropy_scaler = entropy_scaler
        self.l2_reg = l2_reg
        
        self.state_scaler= state_scaler
        self.reward_scaler = reward_scaler

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
            for param in model.parameters():
                weights.append(param.data.view(-1))

            flat_weights = torch.cat(weights)
            weight_norm = torch.norm(flat_weights, 2).item()
            weight_norms.append(weight_norm)
        
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
    
    def normalization(self, input_tuple, update=True):
        """normalize the states, next_states, and rewards if scaler is given
        Args:
            input_tuple: (tuple) (s, a, s', r, m)
            update: (bool) whether the given input is used to update the running_stats
        Returns:
            input_tuple: (tuple) normalized (s, a, s', r, m) 
        """
        states, actions, next_states, rewards, masks = input_tuple
        if self.state_scaler is not None:
            states = self.state_scaler(states, update)
            next_states = self.state_scaler(next_states, update)
        if self.reward_scaler is not None:
            rewards = self.reward_scaler(rewards, update)

        return (states, actions, next_states, rewards, masks)

    def forward(self, input_tuple, deterministic=False):
        """Forward pass used *during sampling*
        Args:
            input_tuple: (tuple) composed of s, a, s', r, m
            deterministic: (bool) boolean for whether deterministic action or not
        Returns:
            action: (1d array) output action
            logprob: (1d tensor) log probability sum for the action
            (y, z): (tuple) computed latent variables 
        """
        input_tuple = self.normalization(input_tuple)
        with torch.no_grad():
            # HL-model; obs and its corresponding categorical inference
            states, y, _, _ = self.HLmodel(input_tuple) 

            # IL-model
            ego_states, z, _, _ = self.ILmodel(states.detach(), y.detach())

            # LL-model
            states = torch.concatenate((ego_states, z), axis=-1)
            action, logprob = self.LLmodel.select_action(states.detach(), deterministic=deterministic)
        
        return action, logprob, (y, z)
        
    def embed(self, input_tuple):
        """Forward pass used *during updating*
        Args:
            input_tuple: (tuple) composed of s, a, s', r, m
        Returns:
            y_embedded_states: (2d array) y augmented states for critic learning
            z_embedded_states: (2d array) z augmented states for actor learning
            (z, z_mu, z_std): (tuple) z information is returned for ILmodel's decoding process
            (loss_cat, loss_occ): (tuple) computed categorical entropy and occupancy loss of y
        """
        # HL
        states, y, loss_cat, loss_occ = self.HLmodel(input_tuple, is_batch=True)

        # IL
        ego_states, z, z_mu, z_std = self.ILmodel(states.detach(), y.detach())

        y_embedded_states = torch.concatenate((states, y), axis=-1) # for critic; s + y
        z_embedded_states = torch.concatenate((ego_states, z), axis=-1) # for actor; s_ego + z = S - s_other + z

        return y_embedded_states, z_embedded_states, (z, z_mu, z_std), (loss_cat, loss_occ)
    
    def learn(self, batch):
        from rlkit.utils.torch import estimate_advantages, estimate_episodic_value
        self.train()
        t_start = time.time()

        # CALL DATA AND PREPROCESS
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        rewards = batch['rewards']
        masks = batch['masks']
        logprobs = batch['logprobs']
        successes = batch['successes']

        mdp_tuple = (states, actions, next_states, rewards, masks)
        
        mdp_tuple = self.normalization(mdp_tuple, update=False)
        states, actions, next_states, rewards, masks = mdp_tuple

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        logprobs = torch.from_numpy(logprobs).to(self.device)
        successes = torch.from_numpy(successes).to(self.device)
        
        # COMPUTE ADVANTAGES FOR POLICY UPDATE
        with torch.no_grad():
            y_embedded_states, _, _, _ = self.embed(mdp_tuple) # for LL actor
            values = self.LLmodel.critic(y_embedded_states)

        # GET THE ACTOR ADVANTAGE ESTIMATION
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)
        episodic_reward = np.sum(batch['rewards'])/ len(np.where(batch['masks'] == 0)[0]) #  estimate_episodic_value(, masks, 1.0, self.device)

        grad_norm_list = []
        weight_norm_list = []
        # Update the parameters
        for _ in range(self.K_epochs):
            y_embedded_states, z_embedded_states, (z, z_mu, z_std), (loss_cat, loss_occ) = self.embed(mdp_tuple) # for LL actor

            # COMPUTE THE CRITIC LOSS THAT TRAINS THE HIGH-LEVEL MODEL AND THE CRITIC ITSELF
            r_pred = self.LLmodel.critic(y_embedded_states) # y embedding b/c it is sub-task info while z is action-task info; this is not action-value fn
            
            value_loss = self.loss_fn(r_pred, returns)
            loss_cat = loss_cat * value_loss.detach() # to scale
            loss_occ = loss_occ * value_loss.detach() # to scale

            # COMPUTE THE VAE (INTERMEDIATE LEVEL) LOSS
            (decoder_loss, state_pred_loss, kl_loss) = self.ILmodel.decode(states, next_states, z, z_mu, z_std)

            # COMPUTE THE ACTOR LOSS
            dist = self.LLmodel.actor(z_embedded_states.detach())

            new_logprobs = dist.log_prob(actions)            
            dist_entropy = dist.entropy()
    
            ratios = torch.exp(new_logprobs - logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            actor_loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * value_loss - self.entropy_scaler * dist_entropy)            
            #print(new_logprobs.shape, logprobs.shape, ratios.shape, surr1.shape, surr2.shape, advantages.shape, dist_entropy.shape, value_loss.shape, decoder_loss.shape)

            # LOSS IS ADDED; THIS IS FINE SINCE THEY ARE CONNECTED TO THE DIFFERENT PARAMETERS
            loss = actor_loss + decoder_loss + loss_cat + loss_occ

            # UPDATE THE ALL MODELS
            self.optimizers.zero_grad()
            loss.backward()

            grad_norm_list.append(self.get_grad_norm())
            weight_norm_list.append(self.get_weight_norm())

            self.optimizers.step()

        training_output = {
            'loss/critic_loss': value_loss.item(),
            'loss/actor_loss': actor_loss.item(),
            'loss/decoder_loss': decoder_loss.item(),
            'loss/decoder_pred_loss': state_pred_loss.item(),
            'loss/decoder_kl_loss': kl_loss.item(),
            'loss/occupancy_loss': loss_occ.item(),
            'loss/cat_ent_loss': loss_cat.item(),
            'train/episodic_reward': episodic_reward,
            'train/success': successes.mean().item(),
        }

        grad_norm = self.average_dict(grad_norm_list)
        param_norm = self.average_dict(weight_norm_list)

        result = {**training_output, **grad_norm, **param_norm}
        
        self.eval()
        t_end = time.time()

        return result, t_end - t_start
    
    def save_model(self, logdir, epoch, is_best=False):
        self.LLmodel = self.LLmodel.cpu()
        self.ILmodel = self.ILmodel.cpu()
        self.HLmodel = self.HLmodel.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump((self.LLmodel, self.ILmodel, self.HLmodel, self.state_scaler, self.reward_scaler), open(path, 'wb'))

        self.LLmodel = self.LLmodel.to(self.device)
        self.ILmodel = self.ILmodel.to(self.device)
        self.HLmodel = self.HLmodel.to(self.device)