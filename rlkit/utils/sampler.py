import random
import time
import math
import os
import torch.multiprocessing as multiprocessing

import torch
import numpy as np
import cv2

from rlkit.utils.functions import visualize_latent_variable
from typing import Optional, Union, Tuple, Dict
from datetime import date

today = date.today()


def allocate_values(total, value):
    result = []
    remaining = total

    while remaining >= value:
        result.append(value)
        remaining -= value

    if remaining != 0:
        result.append(remaining)

    return result


def calculate_workers_and_rounds(environments, episodes_per_env, num_cores):
    if episodes_per_env <= 2:
        num_worker_per_env = 1
    elif episodes_per_env > 2:
        num_worker_per_env = episodes_per_env // 2

    # Calculate total number of workers
    total_num_workers = num_worker_per_env * len(environments)

    if total_num_workers > num_cores:
        avail_core_per_env = num_cores // num_worker_per_env

        num_worker_per_round = allocate_values(
            total_num_workers, avail_core_per_env * num_worker_per_env
        )
        num_env_per_round = allocate_values(len(environments), avail_core_per_env)
        rounds = len(num_env_per_round)
    else:
        num_worker_per_round = [total_num_workers]
        num_env_per_round = [len(environments)]
        rounds = 1

    episodes_per_worker = int(episodes_per_env * len(environments) / total_num_workers)
    return num_worker_per_round, num_env_per_round, episodes_per_worker, rounds


class OnlineSampler:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        episode_len: int,
        episode_num: int,
        training_envs: list,
        running_state=None,
        num_cores: int = None,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.episode_len = episode_len
        self.episode_num = episode_num
        self.training_envs = training_envs
        self.task_names = [env.task_name for env in self.training_envs]
        self.running_state = running_state

        self.device = torch.device(device)

        # Preprocess for multiprocessing to avoid CPU overscription and deadlock
        self.num_cores = (
            num_cores if num_cores is not None else multiprocessing.cpu_count()
        )  # torch.get_num_threads()
        num_workers_per_round, num_env_per_round, episodes_per_worker, rounds = (
            calculate_workers_and_rounds(
                self.training_envs, self.episode_num, self.num_cores
            )
        )

        self.num_workers_per_round = num_workers_per_round
        self.num_env_per_round = num_env_per_round
        self.total_num_worker = sum(self.num_workers_per_round)
        self.episodes_per_worker = episodes_per_worker
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.num_worker_per_env = int(self.total_num_worker / len(self.training_envs))
        self.rounds = rounds

        print("Sampling Parameters:")
        print("--------------------")
        print(
            f"Cores (usage)/(given)             : {self.num_workers_per_round[0]}/{self.num_cores} out of {multiprocessing.cpu_count()}"
        )
        print(f"Number of Environments each Round : {self.num_env_per_round}")
        print(f"Total number of Worker            : {self.total_num_worker}")
        print(f"Episodes per Worker               : {self.episodes_per_worker}")
        torch.set_num_threads(
            1
        )  # enforce one thread for each worker to avoid CPU overscription.

    def get_reset_data(self, batch_size):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        """
        data = dict(
            states=np.zeros((batch_size, self.obs_dim)),
            next_states=np.zeros((batch_size, self.obs_dim)),
            actions=np.zeros((batch_size, self.action_dim)),
            ys=np.zeros((batch_size, self.embed_dim)),
            zs=np.zeros((batch_size, self.action_dim)),
            rewards=np.zeros((batch_size, 1)),
            terminals=np.zeros((batch_size, 1)),
            timeouts=np.zeros((batch_size, 1)),
            masks=np.zeros((batch_size, 1)),
            logprobs=np.zeros((batch_size, 1)),
            successes=np.zeros((batch_size, 1)),
        )
        return data

    def save_rendering(self, recorded_frames, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)
        output_file = os.path.join(path, file_name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI file
        fps = 120
        width = 480
        height = 480
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in recorded_frames:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy,
        thread_batch_size,
        episode_len,
        episode_num,
        seed,
        deterministic=False,
    ):
        # estimate the batch size to hava a large batch
        batch_size = thread_batch_size + episode_len
        data = self.get_reset_data(batch_size=batch_size)
        current_step = 0
        ep_num = 0
        while current_step < thread_batch_size:
            if ep_num >= episode_num:
                break
            # Only for copied worker, apply different seed for stochasticity
            if queue is not None:
                seed = random.randint(1_000, 1_000_000)
                torch.manual_seed(seed)

            # var initialization
            _returns = 0
            max_success = 0.0

            # env initialization
            s, _ = env.reset(seed=seed)

            # create mdp for encoding process. all element should have dimension (1,) than scaler
            a = np.zeros((self.action_dim,))
            ns = s

            input_tuple = (s, a, ns, np.array([0]), np.array([1]))

            # begin the episodic loop by initializing the hidden info of LSTM
            policy.init_encoder_hidden_info()
            for t in range(episode_len):
                # sample action
                with torch.no_grad():
                    a, logprob, (y, z) = policy(
                        input_tuple, deterministic=deterministic
                    )

                # env stepping
                ns, rew, term, trunc, infos = env.step(a)
                success = infos["success"]
                max_success = np.maximum(max_success, success)

                done = trunc or term
                mask = 0 if done else 1

                # state encoding
                input_tuple = (s, a, ns, np.array([rew]), np.array([mask]))

                # saving the data
                data["states"][current_step + t, :] = s
                data["actions"][current_step + t, :] = a
                data["next_states"][current_step + t, :] = ns
                data["ys"][current_step + t, :] = y
                data["zs"][current_step + t, :] = z
                data["rewards"][current_step + t, :] = rew
                data["terminals"][current_step + t, :] = term
                data["timeouts"][current_step + t, :] = trunc
                data["masks"][current_step + t, :] = mask
                data["logprobs"][current_step + t, :] = logprob
                data["successes"][current_step + t, :] = max_success

                s = ns
                _returns += rew

                if done:
                    # clear log
                    ep_num += 1
                    current_step += t + 1
                    _returns = 0
                    break

        memory = dict(
            states=data["states"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            next_states=data["next_states"].astype(np.float32),
            ys=data["ys"].astype(np.float32),
            zs=data["zs"].astype(np.float32),
            rewards=data["rewards"].astype(np.float32),
            terminals=data["terminals"].astype(np.int32),
            timeouts=data["timeouts"].astype(np.int32),
            masks=data["masks"].astype(np.int32),
            logprobs=data["logprobs"].astype(np.float32),
            successes=data["successes"].astype(np.float32),
        )
        if current_step < thread_batch_size:
            for k in memory:
                memory[k] = memory[k][:current_step]
        else:
            for k in memory:
                memory[k] = memory[k][:thread_batch_size]

        # for logging task-wise performance
        task_dict = {
            "train_reward/"
            + env.task_name: np.sum(memory["rewards"])
            / len((np.where(memory["masks"] == 0))[0]),
            "train_success/" + env.task_name: np.mean(memory["successes"]),
        }

        if queue is not None:
            queue.put([pid, memory, task_dict])
        else:
            return memory, task_dict

    def collect_samples(
        self, policy, seed, deterministic=False, pid=None, latent_path=None
    ):
        """
        All sampling and saving to the memory is done in numpy.
        return: dict() with elements in numpy
        """
        t_start = time.time()
        policy.to_device()

        queue = multiprocessing.Manager().Queue()
        env_idx = 0
        worker_idx = 0

        task_dict_list = [None] * self.total_num_worker
        reward_dict = {"train_reward/" + key: 0 for key in self.task_names}
        success_dict = {"train_success/" + key: 0 for key in self.task_names}
        rs_dict = {**reward_dict, **success_dict}

        for round_number in range(self.rounds):
            processes = []
            if round_number == self.rounds - 1:
                envs = self.training_envs[env_idx:]
            else:
                envs = self.training_envs[
                    env_idx : env_idx + self.num_env_per_round[round_number]
                ]
            for env in envs:
                # print(env.task_name)
                workers_for_env = self.num_workers_per_round[round_number] // len(envs)
                for _ in range(workers_for_env):
                    if worker_idx == self.total_num_worker - 1:
                        """Main thread process"""
                        memory, task_dict = self.collect_trajectory(
                            worker_idx,
                            None,
                            env,
                            policy,
                            self.thread_batch_size,
                            self.episode_len,
                            self.episode_num,
                            seed,
                            deterministic,
                        )
                        task_dict_list[-1] = task_dict
                    else:
                        """Sub-thread process"""
                        worker_args = (
                            worker_idx,
                            queue,
                            env,
                            policy,
                            self.thread_batch_size,
                            self.episode_len,
                            self.episode_num,
                            seed,
                            deterministic,
                        )
                        p = multiprocessing.Process(
                            target=self.collect_trajectory, args=worker_args
                        )
                        processes.append(p)
                        p.start()
                    worker_idx += 1
                env_idx += 1
            for p in processes:
                p.join()

        worker_memories = [None] * (worker_idx - 1)
        for _ in range(worker_idx - 1):
            pid, worker_memory, task_dict = queue.get()
            worker_memories[pid] = worker_memory
            task_dict_list[pid] = task_dict

        for worker_memory in worker_memories[::-1]:  # concat in order
            for k in memory:
                memory[k] = np.concatenate((memory[k], worker_memory[k]), axis=0)

        ## for additional logging
        for task_dict in task_dict_list:
            for key, value in task_dict.items():
                rs_dict[key] += value / self.num_worker_per_env

        t_end = time.time()

        if latent_path is not None:
            worker_memories += [memory]
            """draw latent variable !!!"""
            y_info = [
                worker_memories[i]["ys"]
                for i in range(0, len(worker_memories), self.num_worker_per_env)
            ]
            z_info = [
                worker_memories[i]["zs"]
                for i in range(0, len(worker_memories), self.num_worker_per_env)
            ]

            latent_info = [y_info, z_info]

            for info, path in zip(latent_info, latent_path):
                visualize_latent_variable(self.task_names, info, path)

        policy.to_device(self.device)

        return memory, rs_dict, t_end - t_start
