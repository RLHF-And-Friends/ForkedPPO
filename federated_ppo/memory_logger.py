import wandb
import logging
from federated_ppo.utils import safe_wandb_log

class MemoryLogger:
    """
    Class for logging memory information used when exchanging parameters between agents.
    Tracks neural network parameters and policy table parameters, their sizes and data exchange.
    """

    def __init__(self, agent_idx, args, writer, track=False):
        """
        Initialize memory logger

        Args:
            agent_idx: agent index
            args: run arguments
            writer: SummaryWriter for TensorBoard
            track: flag for wandb logging
        """
        self.agent_idx = agent_idx
        self.args = args
        self.writer = writer
        self.track = track

        # Set up logger
        self.logger = logging.getLogger(f"federated_ppo.memory_logger.agent_{self.agent_idx}")

        # Counters for passed/received neural network parameters
        self.total_passed_nn_parameters = 0
        self.total_received_nn_parameters = 0
        self.total_exchanged_nn_parameters = 0

        # Counters for data sizes in megabytes
        self.total_received_fp16_mb = 0
        self.total_passed_fp16_mb = 0
        self.total_exchanged_fp16_mb = 0

        self.total_received_fp32_mb = 0
        self.total_passed_fp32_mb = 0
        self.total_exchanged_fp32_mb = 0

        self.total_received_fp64_mb = 0
        self.total_passed_fp64_mb = 0
        self.total_exchanged_fp64_mb = 0

        # Counters for passed/received policy table parameters
        self.total_passed_policy_table_parameters = 0
        self.total_received_policy_table_parameters = 0
        self.total_exchanged_policy_table_parameters = 0

        # Initialize initial log values
        self.log_initial_memory_info()

    def log_initial_memory_info(self):
        """
        Logs initial information about neural network and policy table sizes
        """
        args = self.args

        # Log to wandb for neural network
        if self.track:
            safe_wandb_log({
                f"exchanged_data_stats/nn_parameters/total_received_fp16_mb/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_passed_fp16_mb/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp16_mb/agent_{self.agent_idx}": 0,

                f"exchanged_data_stats/nn_parameters/total_received_fp32_mb/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_passed_fp32_mb/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp32_mb/agent_{self.agent_idx}": 0,

                f"exchanged_data_stats/nn_parameters/total_received_fp64_mb/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_passed_fp64_mb/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp64_mb/agent_{self.agent_idx}": 0,

                f"exchanged_data_stats/nn_parameters/total_received/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_passed/agent_{self.agent_idx}": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged/agent_{self.agent_idx}": 0,

                f"average_episodic_return/agent_{self.agent_idx}": 0,

                f"exchanged_data_stats/nn_parameters/total_received": 0,
                f"exchanged_data_stats/nn_parameters/total_passed": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged": 0,

                f"exchanged_data_stats/nn_parameters/total_received_fp16_mb": 0,
                f"exchanged_data_stats/nn_parameters/total_passed_fp16_mb": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp16_mb": 0,

                f"exchanged_data_stats/nn_parameters/total_received_fp32_mb": 0,
                f"exchanged_data_stats/nn_parameters/total_passed_fp32_mb": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp32_mb": 0,

                f"exchanged_data_stats/nn_parameters/total_received_fp64_mb": 0,
                f"exchanged_data_stats/nn_parameters/total_passed_fp64_mb": 0,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp64_mb": 0,

                "global_communications": 0,
                "global_step": 0
            })

    def increase_exchanged_nn_parameters(self, number_of_communications, global_step, last_average_episodic_return_between_communications):
        """
        Updates neural network parameter exchange counters and logs information

        Args:
            number_of_communications: current global communication number
            global_step: current global step
        """
        args = self.args

        # Update total parameter counters
        self.total_received_nn_parameters += args.nn_total_params_received_per_global_communication
        self.total_passed_nn_parameters += args.nn_total_params_passed_per_global_communication
        self.total_exchanged_nn_parameters = self.total_received_nn_parameters + self.total_passed_nn_parameters

        fp16_size_mb = 2 / (1024 * 1024)  # 2 bytes -> megabytes
        fp32_size_mb = 4 / (1024 * 1024)  # 4 bytes -> megabytes
        fp64_size_mb = 8 / (1024 * 1024)  # 8 bytes -> megabytes

        received_fp16_mb = args.nn_total_params_received_per_global_communication * fp16_size_mb
        passed_fp16_mb = args.nn_total_params_passed_per_global_communication * fp16_size_mb
        self.total_received_fp16_mb += received_fp16_mb
        self.total_passed_fp16_mb += passed_fp16_mb
        self.total_exchanged_fp16_mb = self.total_received_fp16_mb + self.total_passed_fp16_mb

        received_fp32_mb = args.nn_total_params_received_per_global_communication * fp32_size_mb
        passed_fp32_mb = args.nn_total_params_passed_per_global_communication * fp32_size_mb
        self.total_received_fp32_mb += received_fp32_mb
        self.total_passed_fp32_mb += passed_fp32_mb
        self.total_exchanged_fp32_mb = self.total_received_fp32_mb + self.total_passed_fp32_mb

        received_fp64_mb = args.nn_total_params_received_per_global_communication * fp64_size_mb
        passed_fp64_mb = args.nn_total_params_passed_per_global_communication * fp64_size_mb
        self.total_received_fp64_mb += received_fp64_mb
        self.total_passed_fp64_mb += passed_fp64_mb
        self.total_exchanged_fp64_mb = self.total_received_fp64_mb + self.total_passed_fp64_mb

        # Log to wandb
        if self.track:
            safe_wandb_log({
                # Log total parameter counts
                f"exchanged_data_stats/nn_parameters/total_received/agent_{self.agent_idx}": self.total_received_nn_parameters,
                f"exchanged_data_stats/nn_parameters/total_passed/agent_{self.agent_idx}": self.total_passed_nn_parameters,
                f"exchanged_data_stats/nn_parameters/total_exchanged/agent_{self.agent_idx}": self.total_exchanged_nn_parameters,

                # Log sizes in megabytes for current communication
                f"exchanged_data_stats/nn_parameters/exchanged_fp16_mb/agent_{self.agent_idx}": args.nn_total_params_exchanged_per_global_communication * fp16_size_mb * (number_of_communications + 1),
                f"exchanged_data_stats/nn_parameters/exchanged_fp32_mb/agent_{self.agent_idx}": args.nn_total_params_exchanged_per_global_communication * fp32_size_mb * (number_of_communications + 1),
                f"exchanged_data_stats/nn_parameters/exchanged_fp64_mb/agent_{self.agent_idx}": args.nn_total_params_exchanged_per_global_communication * fp64_size_mb * (number_of_communications + 1),

                # Log accumulated sizes in megabytes
                f"exchanged_data_stats/nn_parameters/total_received_fp16_mb/agent_{self.agent_idx}": self.total_received_fp16_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp16_mb/agent_{self.agent_idx}": self.total_passed_fp16_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp16_mb/agent_{self.agent_idx}": self.total_exchanged_fp16_mb,

                f"exchanged_data_stats/nn_parameters/total_received_fp32_mb/agent_{self.agent_idx}": self.total_received_fp32_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp32_mb/agent_{self.agent_idx}": self.total_passed_fp32_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp32_mb/agent_{self.agent_idx}": self.total_exchanged_fp32_mb,

                f"exchanged_data_stats/nn_parameters/total_received_fp64_mb/agent_{self.agent_idx}": self.total_received_fp64_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp64_mb/agent_{self.agent_idx}": self.total_passed_fp64_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp64_mb/agent_{self.agent_idx}": self.total_exchanged_fp64_mb,

                f"average_episodic_return/agent_{self.agent_idx}": last_average_episodic_return_between_communications,

                f"exchanged_data_stats/nn_parameters/total_received": self.total_received_nn_parameters,
                f"exchanged_data_stats/nn_parameters/total_passed": self.total_passed_nn_parameters,
                f"exchanged_data_stats/nn_parameters/total_exchanged": self.total_exchanged_nn_parameters,

                f"exchanged_data_stats/nn_parameters/total_received_fp16_mb": self.total_received_fp16_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp16_mb": self.total_passed_fp16_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp16_mb": self.total_exchanged_fp16_mb,

                f"exchanged_data_stats/nn_parameters/total_received_fp32_mb": self.total_received_fp32_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp32_mb": self.total_passed_fp32_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp32_mb": self.total_exchanged_fp32_mb,

                f"exchanged_data_stats/nn_parameters/total_received_fp64_mb": self.total_received_fp64_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp64_mb": self.total_passed_fp64_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp64_mb": self.total_exchanged_fp64_mb,

                "global_communications": number_of_communications,
                "global_step": global_step,
            })

    def increase_exchanged_policy_table_parameters(self, number_of_communications, global_step, last_average_episodic_return_between_communications):
        """
        Updates policy table parameter exchange counters and logs information

        Args:
            number_of_communications: current global communication number
            global_step: current global step
        """
        args = self.args

        self.total_received_policy_table_parameters += args.policy_table_total_params_received_per_global_communication
        self.total_passed_policy_table_parameters += args.policy_table_total_params_passed_per_global_communication
        self.total_exchanged_policy_table_parameters = self.total_received_policy_table_parameters + self.total_passed_policy_table_parameters

        # Note: no logs to wandb, it is too large for current environments
        pass

    def log_memory_comparison(self, number_of_communications, global_step):
        """
        Logs comparative memory usage metrics between the policy table and neural network

        Args:
            number_of_communications: current global communication number
            global_step: current global step
        """
        pass
