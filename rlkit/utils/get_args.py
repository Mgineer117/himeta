import argparse


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def get_args():
    parser = argparse.ArgumentParser()
    """WandB and Logging parameters"""
    parser.add_argument(
        "--project", type=str, default="hmrl", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests; if not provided, it will be set automatically.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder; if not provided, it will be set automatically',
    )
    parser.add_argument(
        "--algo-name", type=str, default="ppo", help="additional algo-name for logging"
    )
    parser.add_argument(
        "--log-interval", type=int, default=1, help="logging interval; epoch-based"
    )
    parser.add_argument(
        "--seeds",
        type=list_of_ints,
        default=[1, 3, 5, 7, 9],
        help="--seeds 1,3,5,7,9 # without space",
    )

    """OpenAI Gym parameters"""
    parser.add_argument(
        "--env-type",
        type=str,
        default="MetaGym",
        help="DO NOT CHANGE; MetaWorld is only experimental domains",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="ML10",
        help="Either of MT and ML from MetaWorld",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Used for MT/ML-1 where specified task is needed (pick-place, etc.)",
    )
    parser.add_argument(
        "--task-num",
        type=int,
        default=None,
        help="Used for MT/ML-1 for number of parametric variations (1 ~ n)",
    )

    """Network parameters"""
    # Dimensions can be selected while activations functions are fixed as below:
    # actor, critic, embed-layers => Tanh | encoder, decoder, cat-layer => LeakyReLU
    parser.add_argument("--actor-hidden-dims", type=tuple, default=(256, 256))
    parser.add_argument("--critic-hidden-dims", type=tuple, default=(256, 256))
    parser.add_argument("--encoder-hidden-dims", type=tuple, default=(128, 128, 64, 32))
    parser.add_argument("--decoder-hidden-dims", type=tuple, default=(32, 64, 128, 128))
    parser.add_argument("--categorical-hidden-dims", type=tuple, default=(512, 512))
    parser.add_argument("--recurrent-hidden-size", type=int, default=256)
    parser.add_argument("--state-embed-hidden-dims", type=tuple, default=(64,))
    parser.add_argument("--action-embed-hidden-dims", type=tuple, default=(32,))
    parser.add_argument("--reward-embed-hidden-dims", type=tuple, default=(16,))

    # Learning rates
    parser.add_argument(
        "--actor-lr", type=float, default=3e-4, help="PPO-actor learning rate"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=5e-4, help="PPO-critic learning rate"
    )
    parser.add_argument(
        "--IL-lr",
        type=float,
        default=5e-4,
        help="Intermediate-level model learning rate",
    )
    parser.add_argument(
        "--HL-lr", type=float, default=5e-4, help="High-level model learning rate"
    )
    # PPO parameters
    parser.add_argument(
        "--K-epochs", type=int, default=5, help="PPO update per one iter"
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="clipping parameter for gradient"
    )
    parser.add_argument(
        "--entropy-scaler",
        type=float,
        default=1e-2,
        help="entropy scaler from PPO action-distribution",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.9,
        help="Used in advantage estimation for numerical stability",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount parameters")
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=-0.5,
        help="min deviation as e^sig_min ~= 0.6",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=0.5, help="max deviation as e^sig_max ~= 1.6"
    )

    # Architecutral parameters
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="gru",
        help="gru or lstm for encoder architecture",
    )
    parser.add_argument(
        "--drop-out-rate",
        type=float,
        default=0.7,
        help="used for categorical network and decoder in HL and IL model respectively.",
    )
    parser.add_argument(
        "--occ-loss-type",
        type=str,
        default="exp",
        help="sub-task-wise label occupancy parameters. Either of exp, log, linear or none \
                            It yields penalty as the network wants to use wider range of labels.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=5,
        help="embedding dimension both for categorical network and VAE",
    )
    parser.add_argument(
        "--goal-type",
        type=str,
        default="n_step_forward",
        help="task_subgoal, n_step_forward, fix_by_time",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=3,
        help="How many discrete time steps to forecast; to discover the subgoal that is to be this amount ahead",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="ego",
        help='whether to use masking in VAE; either of "ego" or "none" \
                            ego leaves directly relavant state elements of agent, while none leaves the other rest',
    )
    parser.add_argument(
        "--policy-mask-type",
        type=str,
        default="none",
        help='whether to use masking in LL actor; same as above description; either of "ego" or "none"',
    )
    parser.add_argument(
        "--z-mean-limit", type=tuple, default=(-7.0, 7.0), help="mean clamping for z"
    )
    parser.add_argument(
        "--z-logstd-limit",
        type=tuple,
        default=(-0.5, 0.5),
        help="max deviation clamping for z as e^sig_min ~= 0.6 and e^sig_max ~= 1.6",
    )
    parser.add_argument(
        "--cat-coeff",
        type=float,
        default=1.0,
        help="used for categorical network and decoder in HL and IL model respectively.",
    )
    parser.add_argument(
        "--occ-coeff",
        type=float,
        default=1.0,
        help="used for categorical network and decoder in HL and IL model respectively.",
    )
    parser.add_argument(
        "--value-coeff",
        type=float,
        default=20.0,
        help="used for categorical network and decoder in HL and IL model respectively.",
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=1000,
        help="embedding dimension both for categorical network and VAE",
    )
    parser.add_argument(
        "--num-him-updates",
        type=int,
        default=20,
        help="embedding dimension both for categorical network and VAE",
    )
    parser.add_argument(
        "--reward-bonus",
        type=float,
        default=1.0,
        help="used for categorical network and decoder in HL and IL model respectively.",
    )

    # Conditioner parameters
    parser.add_argument(
        "--IL-condition-y",
        type=bool,
        default=False,
        help="max deviation clamping for z as e^sig_min ~= 0.6 and e^sig_max ~= 1.6",
    )
    parser.add_argument(
        "--IL-condition-z",
        type=bool,
        default=True,
        help="max deviation clamping for z as e^sig_min ~= 0.6 and e^sig_max ~= 1.6",
    )
    parser.add_argument(
        "--LL-condition-y",
        type=bool,
        default=True,
        help="max deviation clamping for z as e^sig_min ~= 0.6 and e^sig_max ~= 1.6",
    )
    parser.add_argument(
        "--LL-condition-z",
        type=bool,
        default=True,
        help="max deviation clamping for z as e^sig_min ~= 0.6 and e^sig_max ~= 1.6",
    )

    """Sampling parameters"""
    parser.add_argument(
        "--epoch",
        type=int,
        default=41,
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--init-epoch",
        type=int,
        default=0,
        help="useful when to resume the previous model training",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=200,
        help="number of iterations within one epoch",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="number of cpu threads to use in sampling; \
                            sampler automatically selects appropriate number of threads given this limit",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=500,
        help="episodic length; useful when one wants to constrain to long to short horizon",
    )
    parser.add_argument(
        "--episode-num",
        type=int,
        default=5,
        help="number of episodes to collect for one env",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )

    """Algorithmic parameters"""
    parser.add_argument(
        "--normalize-state", type=bool, default=True, help="normalise state input"
    )
    parser.add_argument(
        "--normalize-reward", type=bool, default=True, help="normalise reward input"
    )
    parser.add_argument(
        "--reward-conditioner", type=float, default=1e-1, help="reward scaler"
    )
    parser.add_argument(
        "--reward-log-param",
        type=float,
        default=3.0,
        help="from 0 to this param, log is used; remaining is just y=x",
    )
    parser.add_argument(
        "--rendering",
        type=bool,
        default=False,
        help="saves the rendering during evaluation",
    )
    parser.add_argument(
        "--visualize-latent-space",
        type=bool,
        default=True,
        help="saves the latent data into data file and images",
    )
    parser.add_argument(
        "--import-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )
    parser.add_argument("--gpu-idx", type=int, default=0, help="gpu idx to train")
    parser.add_argument("--verbose", type=bool, default=True, help="WandB logging")

    return parser.parse_args()
