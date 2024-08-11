import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def reward_shape(rewards, a):
    idx_below_a = rewards <= a
    idx_above_a = rewards > a

    rewards[idx_below_a] = torch.log( ((np.exp(a) - 1) / a) * rewards[idx_below_a] + 1)
    rewards[idx_above_a] = (a/(a**2 + 1)) * (rewards[idx_above_a] - a) + a

    return rewards

#rewards = torch.rand(size=(20, 1)) * 10
rewards = torch.arange(0, 10, 0.1, dtype=torch.float32)
rewards = rewards[:, None]
a = 3

new_rewards = reward_shape(rewards.clone(), a)

plt.figure(figsize=(14, 12))
plt.plot(rewards, new_rewards)

# Set axis labels
plt.xlabel(r'$R(s_t, a_t)$', fontsize=34)
plt.ylabel(r'$g(R(s_t, a_t))$', fontsize=34)

# Increase tick size
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# Show grid
plt.grid(visible=True)

# Draw a dotted black thick vertical line at a
plt.axvline(x=a, color='black', linestyle='--', linewidth=2)

# Set axis to be equal
plt.axis('scaled')

plt.show()
