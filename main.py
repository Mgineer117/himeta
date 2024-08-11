import uuid
import os
import pickle
import wandb

wandb.require("core")

import torch
from torch.utils.tensorboard import SummaryWriter

from rlkit.utils.torch import seed_all, select_device
from rlkit.utils.functions import get_masking_indices
from rlkit.get_args import get_args
from rlkit.utils.zfilter import ZFilter

from rlkit.nets.hierarchy import LLmodel, ILmodel, HLmodel
from rlkit.nets import HiMeta
from rlkit.utils.load_env import load_metagym_env
from rlkit.utils.sampler import OnlineSampler
from rlkit.utils.buffer import TrajectoryBuffer
from rlkit.utils.wandb_logger import WandbLogger
from rlkit import MFPolicyTrainer


def train(seed):
    args = get_args()
    args.device = select_device(args.gpu_idx)
    args.task = "-".join((args.env_type, args.agent_type))

    if args.group is None:
        args.group = "-".join((args.task, args.algo_name, unique_id))
    if args.name is None:
        args.name = "-".join((args.algo_name, unique_id, "seed:" + str(seed)))
    args.logdir = os.path.join(args.logdir, args.group)

    # seed
    seed_all(seed)

    # create env
    if args.env_type == "MetaGym":
        training_envs, testing_envs = load_metagym_env(args, render_mode="rgb_array")
    else:
        NotImplementedError

    # get dimensional parameters
    args.obs_shape = training_envs[0].observation_space.shape
    args.action_shape = training_envs[0].action_space.shape
    args.max_action = training_envs[0].action_space.high[0]

    # get masking indices; saved in args
    get_masking_indices(args)

    # define sampler for online learning
    sampler = OnlineSampler(
        obs_dim=args.obs_shape[0],
        action_dim=args.action_shape[0],
        embed_dim=args.embed_dim,
        episode_len=args.episode_len,
        episode_num=args.episode_num,
        training_envs=training_envs,
        num_cores=args.num_cores,
        device=args.device,
    )

    buffer = TrajectoryBuffer(
        max_num_trj=args.num_traj,
    )

    # import pre-trained model before defining actual models
    if args.import_model:
        print("Loading previous model parameters....")
        (
            low_level_model,
            int_level_model,
            high_level_model,
            state_scaler,
            reward_scaler,
        ) = pickle.load(open("model/model.p", "rb"))
    else:
        # define running_stat
        state_scaler = ZFilter(shape=args.obs_shape) if args.normalize_state else None
        reward_scaler = args.reward_conditioner if args.normalize_reward else None

        low_level_model = LLmodel(
            actor_hidden_dim=args.actor_hidden_dims,
            critic_hidden_dim=args.critic_hidden_dims,
            state_dim=args.obs_shape[0],
            action_dim=args.action_shape[0],
            latent_dim=args.embed_dim,
            condition_y=args.LL_condition_y,
            condition_z=args.LL_condition_z,
            policy_masking_indices=args.policy_masking_indices,
            max_action=args.max_action,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            device=args.device,
        )

        int_level_model = ILmodel(
            state_dim=args.obs_shape[0],
            action_dim=args.action_shape[0],
            latent_dim=args.embed_dim,
            condition_y=args.IL_condition_y,
            condition_z=args.IL_condition_z,
            z_mean_limit=args.z_mean_limit,
            z_logstd_limit=args.z_logstd_limit,
            goal_type=args.goal_type,
            forecast_steps=args.forecast_steps,
            state_embed_hidden_dims=args.state_embed_hidden_dims,
            encoder_hidden_dims=args.encoder_hidden_dims,
            decoder_hidden_dims=args.decoder_hidden_dims,
            masking_indices=args.masking_indices,
            policy_masking_indices=args.policy_masking_indices,
            drop_out_rate=args.drop_out_rate,
            device=args.device,
        )

        high_level_model = HLmodel(
            state_dim=args.obs_shape[0],
            action_dim=args.action_shape[0],
            latent_dim=args.embed_dim,
            categorical_hidden_dims=args.categorical_hidden_dims,
            recurrent_hidden_size=args.recurrent_hidden_size,
            encoder_type=args.encoder_type,
            state_embed_hidden_dims=args.state_embed_hidden_dims,
            action_embed_hidden_dims=args.action_embed_hidden_dims,
            reward_embed_hidden_dims=args.reward_embed_hidden_dims,
            occ_loss_type=args.occ_loss_type,
            drop_out_rate=args.drop_out_rate,
            device=args.device,
        )

    policy = HiMeta(
        HLmodel=high_level_model,
        ILmodel=int_level_model,
        LLmodel=low_level_model,
        buffer=buffer,
        HL_lr=args.HL_lr,
        IL_lr=args.IL_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        K_epochs=args.K_epochs,
        eps_clip=args.eps_clip,
        reward_log_param=args.reward_log_param,
        cat_coeff=args.cat_coeff,
        occ_coeff=args.occ_coeff,
        value_coeff=args.value_coeff,
        num_him_updates=args.num_him_updates,
        entropy_scaler=args.entropy_scaler,
        state_scaler=state_scaler,
        reward_scaler=reward_scaler,
        reward_bonus=args.reward_bonus,
        device=args.device,
    )

    # setup logger both using WandB and Tensorboard
    default_cfg = vars(args)
    logger = WandbLogger(default_cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(default_cfg, verbose=args.verbose)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        training_envs=training_envs,
        testing_envs=testing_envs,
        sampler=sampler,
        logger=logger,
        writer=writer,
        epoch=args.epoch,
        init_epoch=args.init_epoch,
        step_per_epoch=args.step_per_epoch,
        eval_episodes=args.eval_episodes,
        rendering=args.rendering,
        obs_dim=args.obs_shape[0],
        action_dim=args.action_shape[0],
        embed_dim=args.embed_dim,
        log_interval=args.log_interval,
        visualize_latent_space=args.visualize_latent_space,
        seed=seed,
        device=args.device,
    )

    # begin train
    policy_trainer.train()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    seeds = args.seeds
    unique_id = str(uuid.uuid4())[:4]

    for seed in seeds:
        train(seed)
