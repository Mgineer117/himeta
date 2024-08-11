import pickle
import wandb
import torch
import json



from rlkit.utils.eval_tools import DotDict, evaluate
from rlkit.utils.torch import seed_all
from rlkit.nets import HiMeta
from rlkit.utils.load_env import load_metagym_env
from rlkit.utils.buffer import TrajectoryBuffer
from rlkit.utils.base_logger import BaseLogger

wandb.require("core")



def train(num_episode=10):
    model_dir = 'log/eval_log/model_for_eval/'
    with open(model_dir + 'config.json', 'r') as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    
    # create env
    if args.env_type == "MetaGym":
        training_envs, testing_envs = load_metagym_env(args, render_mode="rgb_array")
    else:
        NotImplementedError

    buffer = TrajectoryBuffer(100)

    # import pre-trained model before defining actual models
    print("Loading previous model parameters....")
    (
        low_level_model,
        int_level_model,
        high_level_model,
        state_scaler,
        reward_scaler,
    ) = pickle.load(open(model_dir + 'model.p', "rb"))
    
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

    # Print the loaded data
    logdir = 'log/eval_log/result/'
    print(f'Saving Directory = {logdir + args.name}')
    print(f'Result is an average of {num_episode} episodes')
    logger = BaseLogger(logdir, name=args.name)

    evaluate(policy, training_envs, testing_envs, logger=logger, num_episode=num_episode)
    print('Evaluation is done!')

    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
