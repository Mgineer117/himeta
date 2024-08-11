import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

import torch.multiprocessing as multiprocessing
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

class DotDict(dict):
    """A dictionary subclass that supports access via the dot operator."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not found in the dictionary")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not found in the dictionary")
        
def evaluate(policy, training_envs, testing_envs, num_episode, logger) -> Dict[str, List[float]]:
    policy.to_device()
    rew_sum = 0; suc_sum = 0

    eval_dict = {}
    envs_list = training_envs + testing_envs

    queue = multiprocessing.Manager().Queue()
    processes = []
    
    for i, env in enumerate(envs_list):
        if i == len(envs_list) - 1:
            '''Main thread process'''
            task_dict, rew_mean, suc_mean = eval_loop(env, policy, num_episode, queue=None)
            eval_dict.update(task_dict)
            rew_sum += rew_mean; suc_sum += suc_mean
        else:
            '''Sub-thread process'''
            p = multiprocessing.Process(target=eval_loop, args=(env, policy, num_episode, queue))
            processes.append(p)
            p.start()

    for p in processes:
        p.join() 
    
    for _ in range(i): 
        task_dict, rew_mean, suc_mean = queue.get()
        eval_dict.update(task_dict)
        rew_sum += rew_mean; suc_sum += suc_mean

    # eval logging
    logger.store(**eval_dict)        
    logger.write(int(0), eval_log = True, display=False)

    avg_rew = rew_sum / len(envs_list)
    avg_suc = suc_sum / len(envs_list)

    return avg_rew, avg_suc

def eval_loop(env, policy, num_episode, queue=None) -> Dict[str, List[float]]:
    torch.set_num_threads(1) # enforce one thread for each worker to avoid CPU overscription.
    eval_ep_info_buffer = []
    for _ in range(num_episode):
        # logging initialization
        max_success = 0.0
        episode_reward, episode_length, episode_success = 0, 0, 0

        # env initialization
        seed = random.randint(10_000, 1_000_000)
        s, _ = env.reset(seed=seed)
        a = np.zeros((4, ))
        ns = s 
        done = False
        input_tuple = (s, a, ns, np.array([0]), np.array([1]))
        
        policy.init_encoder_hidden_info()                
        while not done:
            with torch.no_grad():
                a, _, _ = policy(input_tuple, deterministic=True) #(obs).reshape(1,-1)

            ns, rew, trunc, term, infos = env.step(a.flatten()); success = infos['success']
            done = term or trunc; mask = 0 if done else 1
            max_success = np.maximum(max_success, success)

            episode_reward += rew
            episode_success += max_success
            episode_length += 1
            
            # state encoding
            input_tuple = (s, a, ns, np.array([rew]), np.array([mask]))
            s = ns

            if done:
                eval_ep_info_buffer.append(
                    {"reward": episode_reward, 
                        "success":max_success,
                    }
                )

    task_reward_list = [ep_info["reward"] for ep_info in eval_ep_info_buffer]
    task_success_list = [ep_info["success"] for ep_info in eval_ep_info_buffer]

    task_eval_dict = {
        "reward/" + env.task_name: np.mean(task_reward_list),
        "success/" + env.task_name: np.mean(task_success_list),
    }

    if queue is not None:
        queue.put([task_eval_dict, np.mean(task_reward_list), np.mean(task_success_list)])
    else:
        return task_eval_dict, np.mean(task_reward_list), np.mean(task_success_list)