import time
import os

import random
import gym
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing
import gym
import wandb
from copy import deepcopy

import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from rlkit.utils.sampler import OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.nets import HiMeta

def check_model_params(model_before, model_after):
  """
  This function checks if the parameters of two models are equal.

  Args:
      model_before (torch.nn.Module): Model before the process.
      model_after (torch.nn.Module): Model after the process.

  Returns:
      bool: True if all parameters are equal, False otherwise.
  """
  # Get all parameters from both models
  params_before = list(model_before.parameters())
  params_after = list(model_after.parameters())

  # Check if number of parameters is equal
  if len(params_before) != len(params_after):
    print('not equal')
    return False

  # Check if all corresponding parameters are equal
  for param1, param2 in zip(params_before, params_after):
    if not torch.equal(param1.data, param2.data):
      print('not equal')
      return

  # All parameters are equal
  print('equal')
  return

# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: HiMeta,
        training_envs: List,
        testing_envs: List,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        init_step_per_epoch: int = 0,
        local_steps: int = 3,
        batch_size: int = 256,
        num_trj: int = 0,
        eval_episodes: int = 10,
        rendering: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        obs_dim: int = None,
        action_dim: int = None,
        embed_dim: int = None,
        log_interval: int = 20,
        visualize_latent_space:bool = False,
        seed: int = 0,
        device=torch.device('cpu')
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.training_envs = training_envs
        self.testing_envs = testing_envs
        self.logger = logger
        self.writer = writer

        self._epoch = epoch
        self._init_epoch = init_epoch
        self._step_per_epoch = step_per_epoch
        self._init_step_per_epoch = init_step_per_epoch
        self._local_steps = local_steps
        self._batch_size = batch_size
        self._num_trj = num_trj
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.obs_dim = obs_dim
        self.action_dim = action_dim        
        self.embed_dim = embed_dim
        
        self.last_max_reward = 0.0
        self.last_max_success = 0.0

        self.current_epoch = 0
        self.log_interval = log_interval
        self.rendering = rendering
        self.visualize_latent_space = visualize_latent_space
        self.render_path = None
        self.latent_path = None
        if self.visualize_latent_space:
            directory = os.path.join(self.logger.checkpoint_dir, 'latent')
            os.makedirs(directory)
        self.recorded_frames = []

        self.seed = seed
        self.device = device
    
    def train(self) -> Dict[str, float]:
        start_time = time.time()

        last_3_reward_performance = deque(maxlen=3)
        last_3_success_performance = deque(maxlen=3)
        last_3_final_success_performance = deque(maxlen=3)
        # train loop
        for e in trange(self._init_epoch, self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.recorded_frames = []

            for it in trange(self._init_step_per_epoch, self._step_per_epoch, desc=f"Training", leave=False):
                if e == 0 and it == 0:
                    # first iter evaluate
                    rew_sum, suc_sum, f_suc_sum = self._evaluate(e, it)
                    last_3_reward_performance.append(rew_sum)
                    last_3_success_performance.append(suc_sum)
                    last_3_final_success_performance.append(f_suc_sum)

                if self.visualize_latent_space and self.embed_dim > 0:
                    self.init_latent_path(e, it)
                    self.init_render_path(e, it)

                batch, sample_time = self.sampler.collect_samples(self.policy, render_path=self.render_path, latent_path=self.latent_path)
                loss = self.policy.learn(batch); loss['sample_time'] = sample_time
                # Logging
                self.logger.store(**loss)
                self.logger.write_without_reset(int(e*self._step_per_epoch + it))
                for key, value in loss.items():
                    self.writer.add_scalar(key, value, int(e*self._step_per_epoch + it))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            rew_sum, suc_sum, f_suc_sum = self._evaluate(e, it)
            last_3_reward_performance.append(rew_sum)
            last_3_success_performance.append(suc_sum)
            last_3_final_success_performance.append(f_suc_sum)

            # save checkpoint
            if self.current_epoch % self.log_interval == 0:
                self.policy.save_model(self.logger.checkpoint_dir, e)
            # save the best model
            if np.mean(last_3_reward_performance) >= self.last_max_reward and np.mean(last_3_success_performance) >= self.last_max_success:
                self.policy.save_model(self.logger.log_dir, e, is_best=True)
                self.last_max_reward = np.mean(last_3_reward_performance)
                self.last_max_success = np.mean(last_3_success_performance)
        
        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        self.writer.close()
        return {"last_3_reward_performance": np.mean(last_3_reward_performance),
                "last_3_success_performance": np.mean(last_3_success_performance),
                "last_3_final_success_performance": np.mean(last_3_final_success_performance),
                }

    def _evaluate(self, e, it) -> Dict[str, List[float]]:
        self.policy.eval()
        self.policy.to_device()

        rew_sum = 0; suc_sum = 0; f_suc_sum = 0

        eval_dict = {}
        envs_list = self.training_envs + self.testing_envs

        queue = multiprocessing.Manager().Queue()
        processes = []
        
        for i, env in enumerate(envs_list):
            if i == len(envs_list) - 1:
                '''Main thread process'''
                task_dict, rew_mean, suc_mean, f_suc_mean = self.eval_loop(env, queue=None)
                eval_dict.update(task_dict)
                rew_sum += rew_mean; suc_sum += suc_mean; f_suc_sum += f_suc_mean
            else:
                '''Sub-thread process'''
                p = multiprocessing.Process(target=self.eval_loop, args=(env, queue))
                processes.append(p)
                p.start()

        for p in processes:
            p.join() 

        for _ in range(i - 1): 
            task_dict, rew_mean, suc_mean, f_suc_mean = queue.get()
            eval_dict.update(task_dict)
            rew_sum += rew_mean; suc_sum += suc_mean; f_suc_sum += f_suc_mean

        # eval logging
        self.logger.store(**eval_dict)        
        self.logger.write(int(e*self._step_per_epoch + it), display=False)
        for key, value in eval_dict.items():
            self.writer.add_scalar(key, value, int(e*self._step_per_epoch + it))
        
        self.policy.to_device(self.device)
        self.policy.train()
        
        return rew_sum, suc_sum, f_suc_sum
    
    def eval_loop(self, env, queue=None) -> Dict[str, List[float]]:
        num_episodes = 0
        eval_ep_info_buffer = []
        while num_episodes < self._eval_episodes:
            # initialization
            max_success = 0.0
            s, _ = env.reset(seed=self.seed + num_episodes)
            a = np.zeros((self.action_dim, ))
            ns = s 
            done = False
            input_tuple = (s, a, ns, np.array([0]), np.array([1]))
            episode_reward, episode_length, episode_success, episode_final_success = 0, 0, 0, 0
            self.recorded_frames = []

            self.policy.init_encoder_hidden_info()                
            while not done:
                with torch.no_grad():
                    a, _, _ = self.policy(input_tuple, deterministic=True) #(obs).reshape(1,-1)

                ns, rew, trunc, term, infos = env.step(a.flatten()); success = infos['success']
                done = term or trunc; mask = 0 if done else 1
                max_success = np.maximum(max_success, success)
                
                if self.current_epoch % self.log_interval == 0:
                    if self.rendering and num_episodes == 0:
                        self.recorded_frames.append(env.render())
                
                episode_reward += rew
                episode_success += success
                episode_final_success += max_success
                episode_length += 1
                
                # state encoding
                input_tuple = (s, a, ns, np.array([rew]), np.array([mask]))
                
                s = ns

                if done:
                    if self.current_epoch % self.log_interval == 0:
                        if self.rendering and num_episodes == 0:
                            path = os.path.join(self.logger.checkpoint_dir, 'videos', env.task_name)
                            self.save_rendering(path)

                    eval_ep_info_buffer.append(
                        {env.task_name + "_reward": episode_reward, 
                         env.task_name + "_success":episode_success/episode_length,
                         env.task_name + "_final_success":episode_final_success/episode_length,
                        }
                    )
                    num_episodes +=1
                    episode_reward, episode_length, episode_success, episode_final_success = 0, 0, 0, 0

        task_reward_list = [ep_info[env.task_name + "_reward"] for ep_info in eval_ep_info_buffer]
        task_success_list = [ep_info[env.task_name + "_success"] for ep_info in eval_ep_info_buffer]
        task_final_success_list = [ep_info[env.task_name + "_final_success"] for ep_info in eval_ep_info_buffer]

        task_eval_dict = {
            "eval_reward_mean/" + env.task_name: np.mean(task_reward_list),
            "eval_success_mean/" + env.task_name: np.mean(task_success_list),
            "eval_final_success_mean/" + env.task_name: np.mean(task_final_success_list),
            "eval_reward_std/" + env.task_name: np.std(task_reward_list),
            "eval_success_std/" + env.task_name: np.std(task_success_list),
            "eval_final_success_std/" + env.task_name: np.std(task_final_success_list),
        }

        if np.mean(task_reward_list) >= 5000:
            print('warnming')
            print(np.mean(task_reward_list))
            print(task_reward_list)
            print(task_eval_dict)
            print(task_success_list)

        if queue is not None:
            queue.put([task_eval_dict, np.mean(task_reward_list), np.mean(task_success_list), np.mean(task_final_success_list)])
        else:
            return task_eval_dict, np.mean(task_reward_list), np.mean(task_success_list), np.mean(task_final_success_list)
    
    def save_rendering(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = str(self.current_epoch*self._step_per_epoch) +'.avi'
        output_file = os.path.join(directory, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
        fps = 120
        width = 480
        height = 480
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in self.recorded_frames:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
        self.recorded_frames = []

    def init_latent_path(self, e, it):
        if e % self.log_interval == 0 and it == 0:
            self.latent_path = (os.path.join(self.logger.checkpoint_dir, 'latent', 'y', str(self.current_epoch*self._step_per_epoch) +'.png'),
                                os.path.join(self.logger.checkpoint_dir, 'latent', 'z', str(self.current_epoch*self._step_per_epoch) +'.png'))
        else:
            self.latent_path = None
    def init_render_path(self, e, it):
        if e % self.log_interval == 0 and it == 0:
            self.render_path = os.path.join(self.logger.checkpoint_dir, 'train_video', str(self.current_epoch*self._step_per_epoch))
        else:
            self.render_path = None
