import random
import copy
import time
import gym
import numpy as np
import torch
import concurrent.futures
import torch.optim as optim

from .utils import parse_args, create_comm_matrix
from .agent import Agent
from .federated_environment import FederatedEnvironment

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(gym_id, seed, idx, agent_idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            # See PR: https://github.com/vwxyzjn/ppo-implementation-details/pull/12
            if 'render.modes' not in env.metadata:
                env.metadata['render.modes'] = []
            if 'rgb_array' not in env.metadata['render.modes']:
                env.metadata['render.modes'].append('rgb_array')

            if idx == 0 and agent_idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def exchange_weights(federated_envs) -> None:
    agents = []
    for env in federated_envs:
        agent_copy = copy.deepcopy(env.agent)
        for param in agent_copy.parameters():
            param.requires_grad = False

        agents.append(agent_copy)

    for env in federated_envs:
        env.set_neighbors(agents)


def update_comm_matrix(federated_envs, policy_aggregation_mode) -> None:
    # Note: it could be non-symmetric in case of "average_return"
    assert policy_aggregation_mode == "average_return"

    n = len(federated_envs)
    a = np.zeros(n)
    for i, env in enumerate(federated_envs):
        a[i] = env.last_average_episodic_return_between_communications

    W = np.tile(a, (n, 1))

    for env in federated_envs:
        env.set_comm_matrix(torch.tensor(W, dtype=torch.float32))


def generate_federated_system(device, args, run_name):
    # env setup
    federated_envs = []

    for agent_idx in range(args.n_agents):
        # assumption: seeds < 10
        # so we need to generate (seeds * num_envs * n_agents) different envs
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.gym_id,
                    10 * i * args.n_agents + args.seed * args.n_agents + agent_idx,
                    i,
                    agent_idx,
                    args.capture_video,
                    run_name
                ) for i in range(args.num_envs)
            ]
        )
        
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        federated_envs.append(FederatedEnvironment(device, args, run_name, envs, agent_idx, agent, optimizer))

    if args.use_comm_penalty:
        if args.policy_aggregation_mode == "default":
            comm_matrix = create_comm_matrix(n_agents=args.n_agents, comm_matrix_config=args.comm_matrix_config)
        else:
            comm_matrix = create_comm_matrix(n_agents=args.n_agents, comm_matrix_config=None)
        
        for env in federated_envs:
            env.set_comm_matrix(comm_matrix)
    
    exchange_weights(federated_envs)

    return federated_envs


def local_update(federated_env, number_of_communications) -> None:
    federated_env.local_update(number_of_communications)


if __name__ == "__main__":
    args = parse_args()
    
    if args.use_gym_id_in_run_name:
        run_name = args.gym_id
    else:
        run_name = ""

    if args.exp_name != "":
        if run_name != "":
            run_name += "__"
        run_name += f"{args.exp_name}"

    if args.setup_id != "":
        if run_name != "":
            run_name += "__"
        run_name += f"__{args.setup_id}"

    if args.seed != "":
        run_name += f"__seed_{args.seed}"

    run_name += f"__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
            mode="offline",
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            )
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device: ", device)

    federated_envs = generate_federated_system(device, args, run_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_agents) as executor:
        for number_of_communications in range(0, args.global_updates):
            print("Number of completed communications: ", number_of_communications)
            futures = []
            for i in range(args.n_agents):
                futures.append(executor.submit(local_update, federated_envs[i], number_of_communications))

            for future in futures:
                future.result()

            if args.use_comm_penalty:
                exchange_weights(federated_envs)

            if args.policy_aggregation_mode == "average_return":
                update_comm_matrix(federated_envs, args.policy_aggregation_mode)

            torch.cuda.empty_cache()

    for env in federated_envs:
        env.close() 