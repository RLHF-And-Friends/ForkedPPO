import random
import copy
import time
import datetime
import gym
import numpy as np
import torch
import concurrent.futures
import torch.optim as optim
import argparse
import os
from distutils.util import strtobool

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Agent = None
make_env = None
FederatedEnvironment = None


def average_weights(federated_envs) -> None:
    agents = []
    for env in federated_envs:
        agents.append(copy.deepcopy(env.agent))

    state_dict_keys = agents[0].state_dict().keys()

    for i, env in enumerate(federated_envs):
        agent = env.agent
        averaged_weights = {key: torch.zeros_like(param) for key, param in agents[0].state_dict().items()}
        for key in state_dict_keys:
            denom = 0
            for j, neighbor_agent in enumerate(agents):
                neighbor_agent_weights = neighbor_agent.state_dict()
                averaged_weights[key] += env.comm_matrix[i, j] * neighbor_agent_weights[key]
                denom += env.comm_matrix[i, j]
            averaged_weights[key] /= denom

        agent.load_state_dict(averaged_weights)


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
                    args,
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

    if args.use_comm_penalty or args.average_weights:
        from federated_ppo.utils import create_comm_matrix
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


def add_env_type_arg(parser):
    parser.add_argument("--env-type", type=str, choices=["atari", "minigrid"], default="atari",
        help="Type of environment to use (atari or minigrid)")
    parser.add_argument("--wandb-dir", type=str, default=None,
        help="Directory where wandb logs will be stored. If None, defaults to ROOT_DIR/{env_type}/wandb")
    parser.add_argument("--videos-dir", type=str, default=None,
        help="Directory where videos will be stored. If None, defaults to ROOT_DIR/{env_type}/videos")
    return parser


def main():
    global Agent, make_env, FederatedEnvironment
    
    # Предварительно анализируем аргументы, чтобы узнать тип среды
    parser = argparse.ArgumentParser()
    parser = add_env_type_arg(parser)
    # Добавляем только аргумент для типа среды, чтобы определить, какой модуль импортировать
    temp_args, _ = parser.parse_known_args()
    
    env_type = temp_args.env_type
    
    # Импортируем нужные модули в зависимости от типа среды
    if env_type == "atari":
        print("Используем Atari среду")
        from federated_ppo.atari.agent import Agent as AtariAgent
        from federated_ppo.atari.utils import parse_args, make_env as atari_make_env
        from federated_ppo.federated_environment import FederatedEnvironment as FedEnv
        
        Agent = AtariAgent
        make_env = atari_make_env
        FederatedEnvironment = FedEnv
    else:  # minigrid
        print("Используем MiniGrid среду")
        from federated_ppo.minigrid.agent import Agent as MinigridAgent
        from federated_ppo.minigrid.utils import parse_args, make_env as minigrid_make_env
        from federated_ppo.federated_environment import FederatedEnvironment as FedEnv
        
        Agent = MinigridAgent
        make_env = minigrid_make_env
        FederatedEnvironment = FedEnv
    
    # Теперь можем парсить все аргументы
    args = parse_args()
    
    args.wandb_dir = os.path.join(ROOT_DIR, f"federated_ppo/{args.env_type}/wandb")
    args.videos_dir = os.path.join(ROOT_DIR, f"federated_ppo/{args.env_type}/videos")
    args.runs_dir = os.path.join(ROOT_DIR, f"federated_ppo/{args.env_type}/runs")

    os.makedirs(args.wandb_dir, exist_ok=True)
    os.makedirs(args.videos_dir, exist_ok=True)
    
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

    current_time = datetime.datetime.now().strftime("%d%m_%H%M")
    run_name += f"__{current_time}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            # mode="offline",
            dir=args.wandb_dir,
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

            if args.average_weights:
                average_weights(federated_envs)

            if args.average_weights or args.use_comm_penalty:
                exchange_weights(federated_envs)

            if args.policy_aggregation_mode == "average_return":
                update_comm_matrix(federated_envs, args.policy_aggregation_mode)

            torch.cuda.empty_cache()

    for env in federated_envs:
        env.close()


if __name__ == "__main__":
    main() 