import random
import os
import math
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

def get_masking_indices(args):
    if args.mask_type == "ego":
        masking_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    elif args.mask_type == "other":
        masking_indices = [0, 1, 2, 3, 18, 19, 20, 21]
    elif args.mask_type == "none":
        masking_indices = []
    else:
        NotImplementedError

    if args.policy_mask_type == "ego":
        policy_masking_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    elif args.policy_mask_type == "other":
        policy_masking_indices = [0, 1, 2, 3, 18, 19, 20, 21]
    elif args.policy_mask_type == "none":
        policy_masking_indices = []
    else:
        NotImplementedError
    
    args.masking_indices = masking_indices
    args.policy_masking_indices = policy_masking_indices


def visualize_latent_variable(tasks_name, latent_data, latent_path):
    # Define the task name list
    num_tasks = len(tasks_name)

    if num_tasks > 5:
        col = 2
        row = int(num_tasks/col)
        tasks = [tasks_name[i:i+row] for i in range(0, num_tasks, row)]

    else:
        col = 1
        row = num_tasks
        tasks = [tasks_name]

    data_per_task = 500
    data_dimensions = latent_data[0].shape[-1]
    highlight_interval = 25

    # Generate random data for each task (500, 5) for each task
    data = {task_name: latent_data[i][:data_per_task, :] for i, task_name in enumerate(tasks_name)}

    # save numerical values
    directory, filename = os.path.split(latent_path)
    values_directory = os.path.join(directory, "values")
    
    os.makedirs(values_directory, exist_ok=True)

    data_file_save_directory = os.path.join(values_directory, filename.split(".")[0] + ".h5py")  
    hfile = h5py.File(data_file_save_directory, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')
    hfile.close()

    # Sample every 25th index
    sampled_indices = np.arange(0, data_per_task, highlight_interval)
    num_sampled_points = len(sampled_indices) 

    # Set up the plot with a calculated figure size to ensure square blocks
    fig_width = num_sampled_points * col  # width in "blocks"
    block_size = 1  # each block is a 1x1 square in figure units
    fig_height = row * data_dimensions * block_size

    fig, axs = plt.subplots(row, col, figsize=(fig_width * block_size, fig_height), squeeze=False, sharex=True)

    # Plot each task's data
    for col, task_list in enumerate(tasks):
        for idx, task in enumerate(task_list):
            task_data = data[task]

            # Sample every 25th index
            sampled_data = task_data[sampled_indices]

            # Transpose the data to match the expected heatmap format
            sampled_data_transposed = sampled_data.T

            # Create a heatmap-like plot
            cax = axs[idx, col].imshow(sampled_data_transposed, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
            axs[idx, col].set_title(task, fontsize=32, fontweight='bold')
            axs[idx, col].tick_params(axis='both', which='both', labelsize=18)
            axs[idx, col].set_yticks([])

            # Display numeric values on the heatmap
            for i in range(sampled_data_transposed.shape[1]):
                for j in range(sampled_data_transposed.shape[0]):
                    value = sampled_data[i, j]
                    text = axs[idx, col].text(i, j, f'{value:.2f}', ha='center', va='center', color='black', fontsize=20, fontweight='bold')
                    text.set_path_effects([withStroke(linewidth=2, foreground='white')])

            # Add a color bar to the side
            cbar = fig.colorbar(cax, ax=axs[idx, col])
            cbar.ax.tick_params(labelsize=26)  # Enlarge the ticks and tick labels of the color bar

    # Set the global x-axis labels
    time_intervals = sampled_indices
    for ax in axs[-1, :]:
        ax.set_xticks(np.arange(len(time_intervals)))
        ax.set_xticklabels(time_intervals, fontsize=24)
        ax.set_xlabel('Time', fontsize=28, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(latent_path)
    plt.close()
