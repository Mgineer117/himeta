import argparse

def get_args():
    parser = argparse.ArgumentParser()
    '''WandB and Logging parameters'''
    parser.add_argument("--project", type=str, default="hmrl")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument('--task', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1
    parser.add_argument("--algo-name", type=str, default="ppo")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)

    '''OpenAI Gym parameters'''
    parser.add_argument('--env-type', type=str, default='MetaGym') # Gym or MetaGym
    parser.add_argument('--agent-type', type=str, default='ML10') # MT1, ML45, Hopper, Ant
    parser.add_argument('--task-name', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1 'pick-place'
    parser.add_argument('--task-num', type=int, default=None) # 10, 45, 50

    '''Network parameters'''
    # Dimensions: 
    # actor, critic, embed-layers => Tanh | encoder, decoder, cat-layer => LeakyReLU
    parser.add_argument('--actor-hidden-dims', type=tuple, default=(256, 256))
    parser.add_argument('--critic-hidden-dims', type=tuple, default=(256, 256))
    parser.add_argument('--encoder-hidden-dims', type=tuple, default=(128, 128, 64, 32))
    parser.add_argument('--decoder-hidden-dims', type=tuple, default=(32, 64, 128, 128))
    parser.add_argument('--categorical-hidden-dims', type=tuple, default=(512, 512))
    parser.add_argument('--LSTM-hidden-size', type=int, default=256)
    parser.add_argument('--state-embed-hidden-dims', type=tuple, default=(64, 64))
    parser.add_argument('--action-embed-hidden-dims', type=tuple, default=(32, 32))
    parser.add_argument('--reward-embed-hidden-dims', type=tuple, default=(16, 16))

    # Learning rates
    parser.add_argument("--actor-lr", type=float, default=7e-4)
    parser.add_argument("--critic-lr", type=float, default=7e-4)
    parser.add_argument("--IL-lr", type=float, default=1e-3)
    parser.add_argument("--HL-lr", type=float, default=1e-3)
    # PPO parameters
    parser.add_argument("--K-epochs", type=int, default=5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--entropy-scaler", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--l2-reg", type=float, default=1e-4)

    # Architecutral parameters
    parser.add_argument("--drop-out-rate", type=float, default=0.9)
    parser.add_argument("--occ-loss-type", type=str, default='exp') # exp, log, linear, none
    parser.add_argument("--embed-dim", type=int, default=5)
    parser.add_argument("--mask-type", type=str, default='ego') # ego or other or none # this is for skill embedding
    parser.add_argument("--policy-mask-type", type=str, default='none') # ego or other or none # this is for skill embedding

    '''Sampling parameters'''
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--init-epoch', type=int, default=0)
    parser.add_argument("--step-per-epoch", type=int, default=200)
    parser.add_argument('--num-cores', type=int, default=None)
    parser.add_argument('--episode-len', type=int, default=500)
    parser.add_argument('--episode-num', type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=2)
    
    '''Algorithmic parameters'''
    parser.add_argument("--normalize-state", type=bool, default=False)
    parser.add_argument("--normalize-reward", type=bool, default=False)
    parser.add_argument("--reward-conditioner", type=float, default=1e-2) 
    parser.add_argument("--rendering", type=bool, default=True)
    parser.add_argument("--visualize-latent-space", type=bool, default=True)
    parser.add_argument("--import-model", type=bool, default=False)
    parser.add_argument("--gpu-idx", type=int, default=0)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()