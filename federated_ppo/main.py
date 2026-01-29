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
import pytz
import logging
from typing import List, Optional, Dict, Union, Type, TypeVar, Callable, Any
from federated_ppo.federated_environment import FederatedEnvironment
from federated_ppo.utils import set_nn_and_policy_table_memory_comparison_params
import re

# Create logger for the main module
logger = logging.getLogger("federated_ppo.main")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use Any for type annotation to avoid importing concrete classes
Agent: Optional[Type[Any]] = None
make_env: Optional[Callable] = None
make_minigrid_agent: Optional[Callable] = None  # Variable for the agent creation function

def average_weights(federated_envs: List[FederatedEnvironment], fedavg_average_weights_mode: str) -> None:
    """
    Averages agent weights in a federated system.

    Args:
        federated_envs: List of federated environments with agents
        fedavg_average_weights_mode: Weight averaging mode
            - "classic-average": simple averaging (each agent has equal weight)
            - "weighted-average": averaging using the communication matrix
    """
    n_agents: int = len(federated_envs)

    # Get the state_dict of the first agent to extract keys
    first_agent_state_dict = federated_envs[0].agent.state_dict()
    state_dict_keys = first_agent_state_dict.keys()

    if fedavg_average_weights_mode == "classic-average":
        logger.info("FedAvg average weights mode: classic-average")
        # For classic-average, compute the average once since the result is the same for all agents
        averaged_weights = {key: torch.zeros_like(param) for key, param in first_agent_state_dict.items()}

        # Compute average weights once
        for key in state_dict_keys:
            for env in federated_envs:
                averaged_weights[key] += env.agent.state_dict()[key] / n_agents

        # Load the averaged policy for all agents and assign it to their reference policy
        for env in federated_envs:
            # Update current policy weights
            env.agent.load_state_dict(averaged_weights)

            # Update reference policy weights (equal to current policy)
            with torch.no_grad():
                env.previous_version_of_agent.load_state_dict(averaged_weights)
                for param in env.previous_version_of_agent.parameters():
                    param.requires_grad = False
    else:
        logger.info("FedAvg average weights mode: weighted-average")
        # For weighted mode, individual calculations are needed for each agent
        for i, env in enumerate(federated_envs):
            agent = env.agent
            averaged_weights = {key: torch.zeros_like(param) for key, param in first_agent_state_dict.items()}

            # Weighted averaging using the communication matrix
            for key in state_dict_keys:
                denom: float = 0.0
                for j, env_j in enumerate(federated_envs):
                    averaged_weights[key] += env.comm_matrix[i, j] * env_j.agent.state_dict()[key]
                    denom += env.comm_matrix[i, j]

                # Prevent division by zero
                if denom > 0:
                    averaged_weights[key] /= denom

            # Update current policy weights
            agent.load_state_dict(averaged_weights)

            # Update reference policy weights (equal to current policy)
            with torch.no_grad():
                env.previous_version_of_agent.load_state_dict(averaged_weights)
                for param in env.previous_version_of_agent.parameters():
                    param.requires_grad = False


def exchange_weights(federated_envs: List[FederatedEnvironment], number_of_communications: int) -> None:
    """
    Exchanges weights between agents in a federated system.

    Args:
        federated_envs: List of federated environments with agents
    """
    # Use references to previous_version_of_agent instead of creating new copies
    previous_versions = []
    for env in federated_envs:
        previous_versions.append(env.previous_version_of_agent)

    for env in federated_envs:
        env.set_neighbors(previous_versions)

    update_exchanged_nn_parameters_stats(federated_envs, number_of_communications)


def update_comm_matrix(federated_envs: List[FederatedEnvironment], policy_aggregation_mode: str, fedrl_average_policies_mode: str) -> None:
    """
    Updates the communication matrix based on agent performance.

    Args:
        federated_envs: List of federated environments with agents
        policy_aggregation_mode: Policy aggregation mode
    """
    # Note: it could be non-symmetric in case of "average_return"
    assert policy_aggregation_mode == "average_return"

    n_agents: int = len(federated_envs)
    a: np.ndarray = np.zeros(n_agents)

    if fedrl_average_policies_mode == "weighted-average":
        logger.info("FedRL average policies mode: weighted-average")
        for i, env in enumerate(federated_envs):
            a[i] = env.last_average_episodic_return_between_communications
    elif fedrl_average_policies_mode == "classic-average":
        logger.info("FedRL average policies mode: classic-average")
        for i, env in enumerate(federated_envs):
            a[i] = 1 / n_agents

    W: np.ndarray = np.tile(a, (n_agents, 1))

    for env in federated_envs:
        env.set_comm_matrix(torch.tensor(W, dtype=torch.float32))

def update_exchanged_nn_parameters_stats(federated_envs: List[FederatedEnvironment], number_of_communications: int) -> None:
    for env in federated_envs:
        # Note: we also count the number of parameters exchanged with neighbors with zero communication coefficient
        env.increase_exchanged_nn_parameters_with_neighbors(number_of_communications)
        # env.increase_exchanged_policy_table_parameters_with_neighbors(number_of_communications)
        # env.log_memory_comparison(number_of_communications)


def generate_federated_system(device: torch.device, args: argparse.Namespace, run_name: str) -> List[FederatedEnvironment]:
    """
    Generates a federated training system.

    Args:
        device: Compute device (CPU/GPU)
        args: Command line arguments
        run_name: Experiment run name

    Returns:
        List of federated environments with agents
    """
    # env setup
    federated_envs: List[FederatedEnvironment] = []

    for agent_idx in range(args.n_agents):
        # assumption: seeds < 10
        # so we need to generate (seeds * num_envs * n_agents) different envs
        envs: gym.vector.SyncVectorEnv = gym.vector.SyncVectorEnv(
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

        if args.env_type == "atari" or args.env_type == "minigrid":
            assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        elif args.env_type == "mujoco":
            assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


        # Create agent depending on the environment type
        if args.env_type == "minigrid":
            # Use make_minigrid_agent for minigrid
            agent = make_minigrid_agent(envs, args.agent_with_convolutions).to(device)
        elif args.env_type == "atari":
            # For Atari, use standard creation
            agent = Agent(envs).to(device)
        elif args.env_type == "mujoco":
            agent = Agent(envs).to(device)

        optimizer: optim.Adam = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        federated_envs.append(FederatedEnvironment(device, args, run_name, envs, agent_idx, agent, optimizer))

    set_nn_and_policy_table_memory_comparison_params(args, agent, envs.envs[0], log=True)

    if args.use_comm_penalty or args.use_fedavg:
        from federated_ppo.utils import create_comm_matrix

        if args.policy_aggregation_mode == "default":
            comm_matrix = create_comm_matrix(n_agents=args.n_agents, comm_matrix_config=args.comm_matrix_config)
        else:
            comm_matrix = create_comm_matrix(n_agents=args.n_agents, comm_matrix_config=None)

        for env in federated_envs:
            env.set_comm_matrix(comm_matrix)

    exchange_weights(federated_envs, number_of_communications=1)

    return federated_envs


def local_update(federated_env: FederatedEnvironment, number_of_communications: int) -> None:
    """
    Performs a local update of an agent.

    Args:
        federated_env: Federated environment with an agent
        number_of_communications: Current communication number
    """
    federated_env.local_update(number_of_communications)


def add_env_type_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds arguments for environment type selection.

    Args:
        parser: Argument parser object

    Returns:
        Updated argument parser object
    """
    parser.add_argument("--env-type", type=str, choices=["atari", "minigrid", "mujoco"], default="atari",
        help="Type of environment to use (atari or minigrid)")
    parser.add_argument("--wandb-dir", type=str, default=None,
        help="Directory where wandb logs will be stored. If None, defaults to ROOT_DIR/{env_type}/wandb")
    parser.add_argument("--videos-dir", type=str, default=None,
        help="Directory where videos will be stored. If None, defaults to ROOT_DIR/{env_type}/videos")
    parser.add_argument("--wandb-run-id", type=str, default=None,
        help="Custom ID for wandb run. This will be used as the directory name instead of 'offline-run-DATE_TIME-HASH'")
    return parser


def main() -> None:
    """
    Main function for running federated reinforcement learning.
    """
    global Agent, make_env, make_minigrid_agent

    # Centralized logging setup for the entire project
    root_logger = logging.getLogger("federated_ppo")
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # Pre-parse arguments to determine the environment type
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser = add_env_type_arg(parser)
    # Add only the environment type argument to determine which module to import
    temp_args: argparse.Namespace
    temp_args, _ = parser.parse_known_args()

    env_type: str = temp_args.env_type

    # Import required modules depending on the environment type
    if env_type == "atari":
        logger.info("Using Atari environment")
        from federated_ppo.atari.agent import Agent as AtariAgent
        from federated_ppo.atari.utils import parse_args, make_env as atari_make_env

        Agent = AtariAgent
        make_env = atari_make_env
    elif env_type == "mujoco":
        logger.info("Using Mujoco environment")
        from federated_ppo.mujoco.agent import Agent as MujocoAgent
        from federated_ppo.mujoco.utils import parse_args, make_env as mujoco_make_env

        Agent = MujocoAgent
        make_env = mujoco_make_env
    elif env_type == "minigrid":  # minigrid
        logger.info("Using MiniGrid environment")
        from federated_ppo.minigrid.agent import Agent as MinigridAgent, make_agent
        from federated_ppo.minigrid.utils import parse_args, make_env as minigrid_make_env

        Agent = MinigridAgent
        make_env = minigrid_make_env
        make_minigrid_agent = make_agent  # Initialize the agent creation function

    # Now we can parse all arguments
    args: argparse.Namespace = parse_args()

    args.wandb_dir = os.path.join(ROOT_DIR, f"federated_ppo/{args.env_type}/wandb")
    args.videos_dir = os.path.join(ROOT_DIR, f"federated_ppo/{args.env_type}/videos")
    args.runs_dir = os.path.join(ROOT_DIR, f"federated_ppo/{args.env_type}/runs")

    args.mode = ""
    if args.use_fedavg:
        args.mode = "FedAvg"
    elif args.use_comm_penalty:
        args.mode = "PR"
    else:
        args.mode = "baseline"

    if args.objective_mode == 3:
        args.mode += "-PPO"
    elif args.objective_mode == 4:
        args.mode += "-MDPO"

    if args.use_fedavg:
        if args.fedavg_average_weights_mode == "weighted-average":
            args.mode += "-WeightedAvg"
        elif args.fedavg_average_weights_mode == "classic-average":
            args.mode += "-ClassicAvg"
    elif args.use_comm_penalty:
        if args.fedrl_average_policies_mode == "weighted-average":
            args.mode += "-WeightedAvg"
        elif args.fedrl_average_policies_mode == "classic-average":
            args.mode += "-ClassicAvg"

    os.makedirs(args.wandb_dir, exist_ok=True)
    os.makedirs(args.videos_dir, exist_ok=True)

    run_name: str
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

    utc_plus_3 = pytz.timezone('Europe/Moscow')  # UTC+3 (Moscow time)
    current_time = datetime.datetime.now(utc_plus_3).strftime("%d_%m_%Y_%H_%M_%S")
    run_name += f"__{current_time}"

    safe_run_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", run_name)


    if args.track:
        import wandb
        import os as wandb_os

        wandb_dir = f"federated_ppo/{args.env_type}/wandb/{args.wandb_project_name}"
        os.makedirs(wandb_dir, exist_ok=True)

        wandb_os.environ["WANDB_RUN_ID"] = safe_run_name

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True, # auto-upload the videos of agents playing the game. Note: we have to log videos manually with minigrid environment
            save_code=True,
            mode="offline",
            dir=wandb_dir,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            )
        )
        wandb.config.update({"mode": args.mode})
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    torch.backends.cudnn.benchmark = True  # Improves performance for repeated identical operations

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"device: {device}")

    federated_envs = generate_federated_system(device, args, run_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_agents) as executor:
        for number_of_communications in range(1, args.global_updates + 1):
            logger.info(f"Number of completed communications: {number_of_communications}")
            futures = []
            for i in range(args.n_agents):
                futures.append(executor.submit(local_update, federated_envs[i], number_of_communications))

            for future in futures:
                future.result()

            if args.use_fedavg:
                if args.fedavg_average_weights_mode == "weighted-average":
                    update_comm_matrix(federated_envs, args.policy_aggregation_mode, args.fedavg_average_weights_mode)

                exchange_weights(federated_envs, number_of_communications + 1)
                average_weights(federated_envs, args.fedavg_average_weights_mode)

                # Note: we do not exchange weights, because we do not use neighbors weights in FedAvg between communications
            elif args.use_comm_penalty:
                if args.policy_aggregation_mode == "average_return":
                    update_comm_matrix(federated_envs, args.policy_aggregation_mode, args.fedrl_average_policies_mode)

                exchange_weights(federated_envs, number_of_communications + 1)
            elif args.n_agents == 1:
                exchange_weights(federated_envs, number_of_communications + 1)


            torch.cuda.empty_cache()

        logger.info(f"Number of completed communications: {number_of_communications}")

    for env in federated_envs:
        env.close()


if __name__ == "__main__":
    main()
