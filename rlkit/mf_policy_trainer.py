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
        print("not equal")
        return False

    # Check if all corresponding parameters are equal
    for param1, param2 in zip(params_before, params_after):
        if not torch.equal(param1.data, param2.data):
            print("not equal")
            return

    # All parameters are equal
    print("equal")
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
        eval_episodes: int = 10,
        rendering: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        obs_dim: int = None,
        action_dim: int = None,
        embed_dim: int = None,
        log_interval: int = 2,
        visualize_latent_space: bool = False,
        seed: int = 0,
        device=torch.device("cpu"),
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.training_envs = training_envs
        self.testing_envs = testing_envs
        self.logger = logger
        self.writer = writer

        # training parameters
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._init_epoch = init_epoch
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        # dimensional parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # initialize the essential training components
        self.last_max_reward = 0.0
        self.last_max_success = 0.0
        self.num_env_steps = 0

        # logging parameters
        self.log_interval = log_interval
        self.rendering = rendering
        self.visualize_latent_space = visualize_latent_space
        if self.visualize_latent_space:
            os.makedirs(os.path.join(self.logger.checkpoint_dir, "latent"))
        self.recorded_frames = []

        self.seed = seed
        self.device = device

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        last_3_reward_performance = deque(maxlen=3)
        last_3_success_performance = deque(maxlen=3)
        # train loop
        self.policy.eval()  # policy only has to be train_mode in policy_learn, since sampling needs eval_mode as well.
        for e in trange(self._init_epoch, self._epoch, desc=f"Epoch"):
            self.current_epoch = e

            avg_rew, avg_suc, avg_avg_suc = self.evaluate(e)
            loss = {
                "main_chart/eval_reward": avg_rew,
                "main_chart/eval_success": avg_suc,
                "main_chart/eval_avg_success": avg_avg_suc,
            }
            self.logger.store(**loss)
            self.logger.write(int(e * self._step_per_epoch), display=False)

            last_3_reward_performance.append(avg_rew)
            last_3_success_performance.append(avg_avg_suc)

            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                latent_path = (
                    self.get_latent_path()
                    if self.visualize_latent_space and it == 0
                    else None
                )
                batch, rs_dict, sample_time = self.sampler.collect_samples(
                    self.policy, seed=self.seed, latent_path=latent_path
                )
                loss, update_time = self.policy.learn(batch)
                self.num_env_steps += len(batch["rewards"])

                # Logging further info
                loss = {**loss, **rs_dict}
                loss["main_chart/num_env_steps"] = self.num_env_steps
                loss["main_chart/sample_time"] = sample_time
                loss["main_chart/update_time"] = update_time

                # Logging to WandB and Tensorboard
                self.logger.store(**loss)
                self.logger.write(int(e * self._step_per_epoch + it), display=False)
                for key, value in loss.items():
                    self.writer.add_scalar(
                        key, value, int(e * self._step_per_epoch + it)
                    )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # save checkpoint
            if e % self.log_interval == 0:
                self.policy.save_model(self.logger.checkpoint_dir, e)
            # save the best model
            if (
                np.mean(last_3_reward_performance) >= self.last_max_reward
                and np.mean(last_3_success_performance) >= self.last_max_success
            ):
                self.policy.save_model(self.logger.log_dir, e, is_best=True)
                self.last_max_reward = np.mean(last_3_reward_performance)
                self.last_max_success = np.mean(last_3_success_performance)

        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        self.writer.close()
        wandb.finish()
        return {
            "last_3_reward_performance": np.mean(last_3_reward_performance),
            "last_3_success_performance": np.mean(last_3_success_performance),
        }

    def evaluate(self, e) -> Dict[str, List[float]]:
        self.policy.to_device()

        rew_sum = 0
        suc_sum = 0
        avg_suc_sum = 0

        eval_dict = {}
        envs_list = self.training_envs + self.testing_envs

        if self.rendering:
            """
            Using Multiprocessing crashes graphic rendering process, so we iterate all envs one by one
            """
            for env in envs_list:
                task_dict, rew_mean, suc_mean, avg_suc_mean = self.eval_loop(
                    env, queue=None
                )
                eval_dict.update(task_dict)
                rew_sum += rew_mean
                suc_sum += suc_mean
                avg_suc_sum += avg_suc_mean
        else:
            queue = multiprocessing.Manager().Queue()
            processes = []

            for i, env in enumerate(envs_list):
                if i == len(envs_list) - 1:
                    """Main thread process"""
                    task_dict, rew_mean, suc_mean, avg_suc_mean = self.eval_loop(
                        env, queue=None
                    )
                    eval_dict.update(task_dict)
                    rew_sum += rew_mean
                    suc_sum += suc_mean
                    avg_suc_sum += avg_suc_mean
                else:
                    """Sub-thread process"""
                    p = multiprocessing.Process(
                        target=self.eval_loop, args=(env, queue)
                    )
                    processes.append(p)
                    p.start()

            for p in processes:
                p.join()

            for _ in range(i):
                task_dict, rew_mean, suc_mean, avg_suc_mean = queue.get()
                eval_dict.update(task_dict)
                rew_sum += rew_mean
                suc_sum += suc_mean
                avg_suc_sum += avg_suc_mean

        # eval logging
        self.logger.store(**eval_dict)
        self.logger.write(int(e * self._step_per_epoch), eval_log=True, display=False)
        for key, value in eval_dict.items():
            self.writer.add_scalar(key, value, int(e * self._step_per_epoch))

        self.policy.to_device(self.device)

        avg_rew = rew_sum / len(envs_list)
        avg_suc = suc_sum / len(envs_list)
        avg_avg_suc = avg_suc_sum / len(envs_list)

        return avg_rew, avg_suc, avg_avg_suc

    def eval_loop(self, env, queue=None) -> Dict[str, List[float]]:
        eval_ep_info_buffer = []
        for num_episodes in range(self._eval_episodes):
            # logging initialization
            max_success = 0.0
            self.recorded_frames = []
            episode_reward, episode_length, episode_success = 0, 0, 0
            render_criteria = (
                self.current_epoch % self.log_interval == 0
                and self.rendering
                and num_episodes == 0
            )

            # env initialization
            s, _ = env.reset(seed=self.seed + num_episodes)
            a = np.zeros((self.action_dim,))
            ns = s
            done = False
            input_tuple = (s, a, ns, np.array([0]), np.array([1]))

            self.policy.init_encoder_hidden_info()
            while not done:
                with torch.no_grad():
                    a, _, _ = self.policy(
                        input_tuple, deterministic=True
                    )  # (obs).reshape(1,-1)

                ns, rew, trunc, term, infos = env.step(a.flatten())
                success = infos["success"]
                done = term or trunc
                mask = 0 if done else 1
                max_success = np.maximum(max_success, success)

                episode_reward += rew
                episode_success += max_success
                episode_length += 1

                # state encoding
                input_tuple = (s, a, ns, np.array([rew]), np.array([mask]))
                s = ns

                # render for the video
                if render_criteria:
                    self.recorded_frames.append(env.render())

                if done:
                    if render_criteria:
                        path = os.path.join(
                            self.logger.checkpoint_dir, "videos", env.task_name
                        )
                        self.save_rendering(path)

                    eval_ep_info_buffer.append(
                        {
                            "reward": episode_reward,
                            "success": max_success,
                            "avg_success": episode_success / episode_length,
                        }
                    )

        task_reward_list = [ep_info["reward"] for ep_info in eval_ep_info_buffer]
        task_success_list = [ep_info["success"] for ep_info in eval_ep_info_buffer]
        task_avg_success_list = [
            ep_info["avg_success"] for ep_info in eval_ep_info_buffer
        ]

        task_eval_dict = {
            "reward_mean/" + env.task_name: np.mean(task_reward_list),
            "success_mean/" + env.task_name: np.mean(task_success_list),
            "avg_success_mean/" + env.task_name: np.mean(task_avg_success_list),
            "reward_std/" + env.task_name: np.std(task_reward_list),
            "success_std/" + env.task_name: np.std(task_success_list),
            "avg_success_std/" + env.task_name: np.std(task_avg_success_list),
        }

        if queue is not None:
            queue.put(
                [
                    task_eval_dict,
                    np.mean(task_reward_list),
                    np.mean(task_success_list),
                    np.mean(task_avg_success_list),
                ]
            )
        else:
            return (
                task_eval_dict,
                np.mean(task_reward_list),
                np.mean(task_success_list),
                np.mean(task_avg_success_list),
            )

    def save_rendering(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = str(self.current_epoch * self._step_per_epoch) + ".avi"
        output_file = os.path.join(directory, file_name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI file
        fps = 120
        width = 480
        height = 480
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in self.recorded_frames:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
        self.recorded_frames = []

    def get_latent_path(self):
        latent_path = (
            os.path.join(
                self.logger.checkpoint_dir,
                "latent",
                "y",
                str(self.current_epoch * self._step_per_epoch) + ".png",
            ),
            os.path.join(
                self.logger.checkpoint_dir,
                "latent",
                "z",
                str(self.current_epoch * self._step_per_epoch) + ".png",
            ),
        )
        return latent_path
