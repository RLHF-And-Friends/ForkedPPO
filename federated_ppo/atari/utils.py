import os
import argparse
import numpy as np
import torch
import json
from distutils.util import strtobool
import gym
import torch.nn.functional as F
from typing import Optional
from federated_ppo.federated_environment import compute_kl_divergence
import time

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="",
        help="the name of this experiment")
    parser.add_argument("--setup-id", type=str, default="",
        help="the id of the setup")
    parser.add_argument("--exp-description", type=str, default="Empty description",
        help="Experiment description")
    parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="FedRL_atari",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--use-gym-id-in-run-name", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="run_name format parameter")
    parser.add_argument("--env-type", type=str, default="atari",
        help="Type of environment to use (atari or minigrid)")

    # Algorithm specific arguments
    parser.add_argument("--n-agents", type=int, default=2,
        help="number of agents")
    parser.add_argument("--policy-aggregation-mode", type=str, default="default",
        help="the way we aggregate policies:\n" \
        "1. default (communication matrix)\n" \
        "2. average_return (weigh just neighbors according to their average return between global communications)\n" \
        "3. scalar_product (scalar product of policies)\n"
    )
    parser.add_argument("--local-updates", type=int, default=16,
        help="parameter E (number of local updates between communications)")
    parser.add_argument("--objective-mode", type=int, default=3,
        help="Three modes for objective:\n" \
        "1. No clipping or KL-penalty\n" \
        "2. Clipping\n" \
        "3. KL penalty (fixed or adaptive)\n" \
        "For more details see https://arxiv.org/pdf/1707.06347")
    parser.add_argument("--use-comm-penalty", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Penalize for kl divergence with neighbors or not")
    parser.add_argument("--sum-kl-divergencies", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="In case if --use-comm-penalty=True, sum KL divergencies or KL with weighted distributions")
    parser.add_argument("--comm-penalty-coeff",  type=float, default=1.0,
        help="coefficient of the communication penalty")
    parser.add_argument("--penalty-coeff", type=float, default=1.0,
        help="KL penalty coefficient")
    parser.add_argument("--comm-matrix-config", type=str, default=None, 
        help="path to comm_matrix json-config")
    parser.add_argument("--average-weights", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Average agents weights or not")
    parser.add_argument("--use-fedavg", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use FedAVG or not")
    parser.add_argument(
        "--fedavg-average-weights-mode", type=str, default="classic-avg",
        help="Two modes for FedAVG:\n" \
        "1. classic-avg: just average\n" \
        "2. communication-avg: average with communication matrix\n"
    )

    # PPO specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.global_updates = int(int(args.total_timesteps // args.batch_size) // args.local_updates)
    print("Expected number of communications in total: ", args.global_updates)
    print("Local updates between communications: ", args.batch_size * args.local_updates)

    assert args.objective_mode in [2, 3, 4]
    assert args.policy_aggregation_mode in ["default", "average_return", "scalar_product"]
    assert args.fedavg_average_weights_mode in ["classic-avg", "communication-avg"]
    assert not (args.use_fedavg and args.use_comm_penalty)

    # fmt: on
    return args

def make_env(args, gym_id, seed, idx, agent_idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        if capture_video:
            # See PR: https://github.com/vwxyzjn/ppo-implementation-details/pull/12
            if 'render.modes' not in env.metadata:
                env.metadata['render.modes'] = []
            if 'rgb_array' not in env.metadata['render.modes']:
                env.metadata['render.modes'].append('rgb_array')

            if idx == 0 and agent_idx == 0:
                env = gym.wrappers.RecordVideo(env, os.path.join(args.videos_dir, f"env_{agent_idx}/{run_name}"))

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
