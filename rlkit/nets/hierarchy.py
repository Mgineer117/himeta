import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional

from rlkit.nets.rnn import RecurrentEncoder
from rlkit.nets.mlp import MLP
from rlkit.modules import ActorProb, Critic, DiagGaussian


class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim, device):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim).to(device)
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.device = device

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y.squeeze()


class LLmodel(nn.Module):
    def __init__(
        self,
        actor_hidden_dim: tuple,
        critic_hidden_dim: tuple,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        condition_y: bool = True,
        condition_z: bool = True,
        policy_masking_indices: List = [],
        max_action: int = 1.0,
        sigma_min: float = -5.0,
        sigma_max: float = 2.0,
        device=torch.device("cpu"),
    ):
        super(LLmodel, self).__init__()  # - len(masking_indices)
        actor_input_dim = (
            int(condition_y) * latent_dim
            + int(condition_z) * action_dim
            + state_dim
            - len(policy_masking_indices)
        )

        actor_backbone = MLP(
            input_dim=actor_input_dim,
            hidden_dims=actor_hidden_dim,
            activation=torch.nn.Tanh,
        )
        critic_backbone = MLP(
            input_dim=latent_dim + state_dim,
            hidden_dims=critic_hidden_dim,
            activation=torch.nn.Tanh,
        )

        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=False,
            conditioned_sigma=True,
            max_mu=max_action,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        actor = ActorProb(actor_backbone, dist_net=dist, device=device)

        critic = Critic(critic_backbone, device=device)

        self.actor = actor
        self.critic = critic

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.condition_y = condition_y
        self.condition_z = condition_z

        self.param_size = sum(p.numel() for p in self.actor.parameters())
        self.device = device

    def change_device_info(self, device):
        self.actor.device = device
        self.critic.device = device
        self.device = device

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        logprob = dist.log_prob(action)
        return action, logprob

    def select_action(self, obs, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action, logprob = self.actforward(obs, deterministic)
        return action.cpu().numpy(), logprob.cpu().numpy()


class ILmodel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        condition_y: bool = True,
        condition_z: bool = True,
        z_mean_limit: tuple = (-3.0, 3.0),
        z_logstd_limit: tuple = (-0.5, 0.5),
        goal_type: str = "n_step_forward",
        forecast_steps: int = 15,
        state_embed_hidden_dims: tuple = (64, 64),
        encoder_hidden_dims: tuple = (128, 64, 32),
        decoder_hidden_dims: tuple = (32, 64, 128),
        masking_indices: List = [],
        policy_masking_indices: List = [],
        drop_out_rate: float = 0.7,
        device=torch.device("cpu"),
    ) -> None:
        super(ILmodel, self).__init__()
        # save parameter first
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.condition_y = condition_y
        self.condition_z = condition_z
        self.z_mean_min = z_mean_limit[0]
        self.z_mean_max = z_mean_limit[1]
        self.z_logstd_min = z_logstd_limit[0]
        self.z_logstd_max = z_logstd_limit[1]
        self.goal_type = goal_type
        self.forecast_steps = forecast_steps
        self.masking_indices = masking_indices
        self.masking_length = len(self.masking_indices)
        self.policy_masking_indices = policy_masking_indices
        self.drop_out_rate = drop_out_rate
        self.device = device

        """Define model"""
        # embedding has tanh as an activation function while encoder and decoder have ReLU
        self.pre_embed = MLP(
            input_dim=state_dim - self.masking_length,
            hidden_dims=(state_dim - self.masking_length,),
            activation=nn.Tanh,
            device=device,
        )

        self.encoder = MLP(
            input_dim=state_dim - self.masking_length + latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=nn.ReLU,
            device=device,
        )

        self.mu_network = nn.Linear(encoder_hidden_dims[-1], action_dim).to(device)
        self.logstd_network = nn.Linear(encoder_hidden_dims[-1], action_dim).to(device)

        self.post_embed = MLP(
            input_dim=state_dim - self.masking_length,
            hidden_dims=(state_dim - self.masking_length,),
            activation=nn.Tanh,
            device=device,
        )

        decoder_input_dim = (
            int(condition_y) * latent_dim
            + int(condition_z) * action_dim
            # + state_dim
            # - self.masking_length
        )

        self.decoder = MLP(
            input_dim=decoder_input_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=state_dim - self.masking_length,
            activation=nn.ReLU,
            dropout_rate=self.drop_out_rate,
            device=device,
        )

        self.to(device=self.device)

    def change_device_info(self, device):
        self.pre_embed.device = device
        self.mu_network.device = device
        self.logstd_network.device = device
        self.encoder.device = device
        self.decoder.device = device
        self.device = device

    def forward(
        self,
        states: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embedding network
        states = self.torch_delete(states, self.masking_indices, axis=-1)
        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)

        states = self.pre_embed(states)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)

        input = torch.concatenate((states, y), axis=-1)

        # encoder
        logits = self.encoder(input)

        z_mu = self.mu_network(logits)
        z_std = self.logstd_network(logits)

        z_std = torch.exp(
            torch.clamp(z_std, self.z_logstd_min, self.z_logstd_max)
        )  # clamping b/w -5 and 2

        dist = torch.distributions.MultivariateNormal(z_mu, torch.diag_embed(z_std))
        z = dist.rsample()
        z_entropy = torch.mean(dist.entropy())

        return z, z_mu, z_std, z_entropy, dist

    def preprocess4forcast(self, ego_next_states, y, masks):
        n = (
            self.forecast_steps - 1
        )  # -1 because next-state is already t + 1 step forward than state
        rows, cols = ego_next_states.shape
        new_ego_next_states = torch.zeros((rows, cols)).to(self.device)

        prev_idx = 0
        mask_idx = torch.where(masks == 0)[0]
        if self.goal_type == "task_subgoal":
            """
            Set the goal when the task-label changes
            """
            y = torch.argmax(y, dim=-1)
            changing_indices = (y[:-1] != y[1:]).nonzero(as_tuple=True)[0] + 1
            for idx in mask_idx:
                boolean = torch.logical_and(
                    changing_indices >= prev_idx, changing_indices <= idx
                )
                if len(changing_indices[boolean]) != 0:
                    for ep_idx in changing_indices[boolean]:
                        new_ego_next_states[prev_idx:ep_idx, :] = ego_next_states[
                            ep_idx - 1, :
                        ]
                        prev_idx = ep_idx

                    new_ego_next_states[prev_idx : ep_idx + 1, :] = ego_next_states[
                        idx, :
                    ]
                    prev_idx = ep_idx + 1

                else:
                    # in case, there is no change in y
                    new_ego_next_states[prev_idx : idx + 1, :] = ego_next_states[
                        prev_idx : idx + 1, :
                    ]
                    prev_idx = idx + 1

        elif self.goal_type == "n_step_forward":
            """
            Set the goal as the state that is n_step forward
            """
            for idx in mask_idx:
                new_ego_next_states[prev_idx : idx - n + 1, :] = ego_next_states[
                    prev_idx + n : idx + 1, :
                ]
                new_ego_next_states[idx - n + 1 : idx + 1, :] = ego_next_states[idx, :]

                prev_idx = idx + 1
        elif self.goal_type == "fix_by_time":
            """
            Fix the goal to the specified time interval
            """
            for idx in mask_idx:
                for ep_idx in range(prev_idx, idx - n, n):
                    new_ego_next_states[prev_idx : ep_idx + n, :] = ego_next_states[
                        ep_idx + n, :
                    ]
                    prev_idx = ep_idx + n
                new_ego_next_states[prev_idx : idx + 1, :] = ego_next_states[idx, :]
                prev_idx = idx + 1

        return new_ego_next_states

    def find_elevated_sum(self, actions, y, masks):
        y = torch.argmax(y, dim=-1)
        task_idx = (y[:-1] != y[1:]).nonzero(as_tuple=True)[0] + 1
        mask_idx = torch.where(masks == 0)[0]

        elevated_sum = torch.zeros_like(actions)

        prev_idx = 0
        for m_idx in mask_idx:
            boolean = torch.logical_and(task_idx >= prev_idx, task_idx <= m_idx)
            if len(task_idx[boolean]) != 0:
                for y_idx in task_idx[boolean]:
                    elevated_sum[prev_idx] = actions[prev_idx]
                    for t in range(prev_idx + 1, y_idx + 1):
                        elevated_sum[t] = elevated_sum[t - 1] + actions[t]
                    prev_idx = y_idx + 1

                if prev_idx - m_idx <= -1:
                    elevated_sum[prev_idx] = actions[prev_idx]
                    if prev_idx + 1 - m_idx <= -1:
                        for t in range(prev_idx + 1, m_idx + 1):
                            elevated_sum[t] = elevated_sum[t - 1] + actions[t]
                prev_idx = m_idx + 1
            else:
                elevated_sum[prev_idx] = actions[prev_idx]
                for t in range(prev_idx + 1, m_idx + 1):
                    elevated_sum[t] = elevated_sum[t - 1] + actions[t]
                prev_idx = m_idx + 1

        return elevated_sum

    def decode(
        self,
        y: torch.Tensor,
        masks: torch.Tensor,
        next_states: torch.Tensor,
        z: Optional[torch.Tensor],
        z_dist: torch.distributions,
        a_dist: torch.distributions,
    ) -> torch.Tensor:
        ego_next_states = self.torch_delete(next_states, self.masking_indices, axis=-1)
        ego_next_states = self.post_embed(ego_next_states)

        new_ego_next_states = self.preprocess4forcast(
            ego_next_states, y.clone().detach(), masks
        )

        next_state_pred = self.decoder(z)

        state_pred_loss = F.mse_loss(next_state_pred, new_ego_next_states)
        kl_loss = torch.mean(torch.distributions.kl_divergence(z_dist, a_dist))

        action_pred_loss = torch.tensor(0.0)

        ELBO_loss = state_pred_loss + kl_loss  #

        return (ELBO_loss, state_pred_loss, kl_loss, action_pred_loss)

    def torch_delete(self, tensor, indices, axis=None):
        tensor = tensor.cpu().numpy()
        tensor = np.delete(tensor, indices, axis=axis)
        tensor = torch.tensor(tensor).to(self.device)
        return tensor


# MetaWorld Gaussian Mixture Variational Auto-Encoder
class HLmodel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        categorical_hidden_dims: tuple = (512, 512),
        recurrent_hidden_size: int = 256,
        encoder_type: str = "gru",
        state_embed_hidden_dims: tuple = (64, 64),
        action_embed_hidden_dims: tuple = (64, 64),
        reward_embed_hidden_dims: tuple = (64, 64),
        occ_loss_type: str = "exp",
        drop_out_rate: float = 0.7,
        device=torch.device("cpu"),
    ) -> None:
        super(HLmodel, self).__init__()
        """parameter save to the class"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.occ_loss_type = occ_loss_type
        self.drop_out_rate = drop_out_rate
        self.device = device

        feature_dim = state_dim + action_dim + state_dim + 1

        """Pre-embedding"""
        self.state_embed = MLP(
            input_dim=state_dim,
            hidden_dims=state_embed_hidden_dims + (state_dim,),
            activation=nn.Tanh,
            device=device,
        )
        self.action_embed = MLP(
            input_dim=action_dim,
            hidden_dims=action_embed_hidden_dims + (action_dim,),
            activation=nn.Tanh,
            device=device,
        )
        self.reward_embed = MLP(
            input_dim=1,
            hidden_dims=reward_embed_hidden_dims + (1,),
            activation=nn.Tanh,
            device=device,
        )

        """Encoder definitions"""
        self.encoder = RecurrentEncoder(
            input_size=feature_dim,
            hidden_size=recurrent_hidden_size,
            encoder_type=encoder_type,
            device=device,
        )
        # cat(h) -> y -> en(h, y)
        self.cat_layer = MLP(
            input_dim=recurrent_hidden_size,
            hidden_dims=categorical_hidden_dims,  # hidden includes the relu activation
            activation=nn.ReLU,
            dropout_rate=self.drop_out_rate,
            device=device,
        )

        self.Gumbel_layer = GumbelSoftmax(
            categorical_hidden_dims[-1], self.latent_dim, device
        )

        self.to(device=self.device)

    def change_device_info(self, device):
        self.state_embed.device = device
        self.action_embed.device = device
        self.reward_embed.device = device
        self.encoder.device = device
        self.cat_layer.device = device
        self.Gumbel_layer.device = device
        self.device = device

    def pack4rnn(self, input_tuple, is_batch):
        """
        Input: tuple of s, a, ns, r, m
        Return: padded_data (batch, seq, fea) and legnths for each trj
        =============================================
        1. find the maximum length of the given traj
        2. create a initialized batch with that max traj length
        3. put the values in
        4. return the padded data and its corresponding length for later usage.
        """
        obss, actions, next_obss, rewards, masks = input_tuple
        if is_batch:
            trajs = []
            lengths = []
            prev_i = 0
            for i, mask in enumerate(masks):
                if mask == 0:
                    trajs.append(
                        torch.concatenate(
                            (
                                obss[prev_i : i + 1, :],
                                actions[prev_i : i + 1, :],
                                next_obss[prev_i : i + 1, :],
                                rewards[prev_i : i + 1, :],
                            ),
                            axis=-1,
                        )
                    )
                    lengths.append(i + 1 - prev_i)
                    prev_i = i + 1

            # pad the data
            largest_length = max(lengths)
            data_dim = trajs[0].shape[-1]
            padded_data = torch.zeros((len(lengths), largest_length, data_dim))

            for i, traj in enumerate(trajs):
                padded_data[i, : lengths[i], :] = traj

            return (padded_data, lengths)
        else:
            states, actions, next_states, rewards, masks = input_tuple
            mdp = torch.concatenate((states, actions, next_states, rewards), axis=-1)
            # convert to 3d aray for rnn
            mdp = mdp[None, None, :]
            return (mdp, None)

    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)
        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def occupancy_loss(self, y):
        """occupancy loss to suppress usage of larger subtask index
            loss = logK * (1/f(K)) * -Σ (1,...,f(K)) dot y for linear
        Args:
            y: sampled subtask composition
        args.type: How f(K) is defined, linear square, or exponential
        maximum is set as log(K) to match magnitude of the entropy loss
        """
        if self.occ_loss_type == "linear":
            occ_coeff = (
                np.log(self.latent_dim)
                * torch.arange(1.0, self.latent_dim + 1)
                / self.latent_dim
            )
        elif self.occ_loss_type == "log":
            occ_coeff = torch.log(torch.arange(1.0, self.latent_dim + 1))
        elif self.occ_loss_type == "exp":
            occ_coeff = (
                np.log(self.latent_dim)
                * torch.exp(torch.arange(1.0, self.latent_dim + 1))
                / np.exp(self.latent_dim)
            )
        elif self.occ_loss_type == "none":
            return torch.tensor(0.0)

        occ_loss = occ_coeff.to(self.device) * y

        return torch.mean(torch.sum(occ_loss, dim=-1))

    def forward(
        self,
        input_tuple: tuple,
        is_batch: bool = False,
    ) -> Tuple[torch.Tensor]:
        states, actions, next_states, rewards, masks = input_tuple

        # conversion
        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(
            next_states, device=self.device, dtype=torch.float32
        )
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.int32)

        # print(states.shape, actions.shape, next_states.shape, rewards.shape, masks.shape)
        """Embedding"""
        states = self.state_embed(states)
        actions = self.action_embed(actions)
        next_states = self.state_embed(next_states)
        rewards = self.reward_embed(rewards)

        input_tuple = (states, actions, next_states, rewards, masks)

        mdp_and_lengths = self.pack4rnn(input_tuple, is_batch)
        out = self.encoder(mdp_and_lengths, is_batch)

        # categorical
        out = self.cat_layer(out)
        logits, prob, y = self.Gumbel_layer(out)
        loss_cat = -self.entropy(logits, prob) - np.log(
            1.0 / self.latent_dim
        )  # uniform entropy
        loss_occ = self.occupancy_loss(y)

        return y, loss_cat, loss_occ  # this pair directly goes to IL
