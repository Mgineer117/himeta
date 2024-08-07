import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from typing import Optional, Union, Tuple, Dict, List

class TrajectoryBuffer:
    def __init__(
        self,
        max_num_trj: int,
        device: str = torch.device('cpu')
    ) -> None:
        self.max_num_trj = max_num_trj
        self.device = device
        
        # Using lists to store trajectories
        self.trajectories = []
        self.num_trj = 0
    
    def decompose(self, states, actions, next_states, rewards, masks) -> List[Dict[str, np.ndarray]]:
        trajs = []
        prev_i = 0
        for i, mask in enumerate(masks.squeeze()):
            if mask == 0:
                data = {
                    "states": states[prev_i:i+1],
                    "actions": actions[prev_i:i+1],
                    "next_states": next_states[prev_i:i+1],
                    "rewards": rewards[prev_i:i+1],
                    "masks": masks[prev_i:i+1]
                }
                trajs.append(data)
                prev_i = i + 1
        return trajs

    def push(self, batch: dict) -> None:
        state, action, next_state, reward, mask = \
            batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['masks']
        
        trajs = self.decompose(state, action, next_state, reward, mask)

        for traj in trajs:
            if self.num_trj < self.max_num_trj:
                self.trajectories.append(traj)
            else:
                self.trajectories[self.num_trj % self.max_num_trj] = traj
            self.num_trj += 1

    def sample(self, num_traj: int) -> Dict[str, torch.Tensor]:
        if num_traj > self.num_trj:
            num_traj = self.num_trj
        
        # Sample random trajectories
        sampled_indices = np.random.choice(min(self.num_trj, self.max_num_trj), num_traj, replace=False)
        
        # Collect sampled data and concatenate
        sampled_data = [self.trajectories[idx] for idx in sampled_indices]
        
        sampled_batch = {
            'states': np.concatenate([traj['states'] for traj in sampled_data], axis=0),
            'actions': np.concatenate([traj['actions'] for traj in sampled_data], axis=0),
            'next_states': np.concatenate([traj['next_states'] for traj in sampled_data], axis=0),
            'rewards': np.concatenate([traj['rewards'] for traj in sampled_data], axis=0),
            'masks': np.concatenate([traj['masks'] for traj in sampled_data], axis=0)
        }

        return sampled_batch
