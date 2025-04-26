import wandb

class MemoryLogger:
    """
    Класс для логирования информации о памяти, используемой при обмене параметрами между агентами.
    Отслеживает параметры нейронной сети и таблицы политики, их размеры и обмен данными.
    """
    
    def __init__(self, agent_idx, args, writer, track=False):
        """
        Инициализация логгера памяти
        
        Args:
            agent_idx: индекс агента
            args: аргументы запуска
            writer: SummaryWriter для TensorBoard
            track: флаг для логирования в wandb
        """
        self.agent_idx = agent_idx
        self.args = args
        self.writer = writer
        self.track = track
        
        # Счетчики переданных/полученных параметров нейронной сети
        self.total_passed_nn_parameters = 0
        self.total_received_nn_parameters = 0
        self.total_exchanged_nn_parameters = 0
        
        # Счетчики для размеров данных в мегабайтах
        self.total_received_fp16_mb = 0
        self.total_passed_fp16_mb = 0
        self.total_exchanged_fp16_mb = 0
        
        self.total_received_fp32_mb = 0
        self.total_passed_fp32_mb = 0
        self.total_exchanged_fp32_mb = 0
        
        self.total_received_fp64_mb = 0
        self.total_passed_fp64_mb = 0
        self.total_exchanged_fp64_mb = 0
        
        # Счетчики переданных/полученных параметров таблицы политики
        self.total_passed_policy_table_parameters = 0
        self.total_received_policy_table_parameters = 0
        self.total_exchanged_policy_table_parameters = 0
        
        # Инициализация начальных значений в логах
        self.log_initial_memory_info()
    
    def log_initial_memory_info(self):
        """
        Логирует исходную информацию о размерах нейронной сети и таблицы политики
        """
        args = self.args
                
        # Логирование в wandb для нейронной сети
        if self.track:
            wandb.log({
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
                
                "global_communications": 0,
                "global_step": 0
            })
    
    def increase_exchanged_nn_parameters(self, number_of_communications, global_step):
        """
        Обновляет счетчики обмена параметрами нейронной сети и логирует информацию
        
        Args:
            number_of_communications: номер текущей глобальной коммуникации
            global_step: текущий глобальный шаг
        """
        args = self.args
        
        # Обновляем общие счетчики параметров
        self.total_received_nn_parameters += args.nn_total_params_received_per_global_communication
        self.total_passed_nn_parameters += args.nn_total_params_passed_per_global_communication
        self.total_exchanged_nn_parameters = self.total_received_nn_parameters + self.total_passed_nn_parameters
        
        fp16_size_mb = 2 / (1024 * 1024)  # 2 байта -> мегабайты
        fp32_size_mb = 4 / (1024 * 1024)  # 4 байта -> мегабайты
        fp64_size_mb = 8 / (1024 * 1024)  # 8 байт -> мегабайты
        
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
                                      
        # Логирование в wandb
        if self.track:
            wandb.log({
                # Логирование общего количества параметров
                f"exchanged_data_stats/nn_parameters/total_received/agent_{self.agent_idx}": self.total_received_nn_parameters,
                f"exchanged_data_stats/nn_parameters/total_passed/agent_{self.agent_idx}": self.total_passed_nn_parameters,
                f"exchanged_data_stats/nn_parameters/total_exchanged/agent_{self.agent_idx}": self.total_exchanged_nn_parameters,
                
                # Логирование размеров в мегабайтах для текущей коммуникации
                f"exchanged_data_stats/nn_parameters/exchanged_fp16_mb/agent_{self.agent_idx}": args.nn_total_params_exchanged_per_global_communication * fp16_size_mb * (number_of_communications + 1),
                f"exchanged_data_stats/nn_parameters/exchanged_fp32_mb/agent_{self.agent_idx}": args.nn_total_params_exchanged_per_global_communication * fp32_size_mb * (number_of_communications + 1),
                f"exchanged_data_stats/nn_parameters/exchanged_fp64_mb/agent_{self.agent_idx}": args.nn_total_params_exchanged_per_global_communication * fp64_size_mb * (number_of_communications + 1),
                
                # Логирование накопленных размеров в мегабайтах
                f"exchanged_data_stats/nn_parameters/total_received_fp16_mb/agent_{self.agent_idx}": self.total_received_fp16_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp16_mb/agent_{self.agent_idx}": self.total_passed_fp16_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp16_mb/agent_{self.agent_idx}": self.total_exchanged_fp16_mb,
                
                f"exchanged_data_stats/nn_parameters/total_received_fp32_mb/agent_{self.agent_idx}": self.total_received_fp32_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp32_mb/agent_{self.agent_idx}": self.total_passed_fp32_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp32_mb/agent_{self.agent_idx}": self.total_exchanged_fp32_mb,
                
                f"exchanged_data_stats/nn_parameters/total_received_fp64_mb/agent_{self.agent_idx}": self.total_received_fp64_mb,
                f"exchanged_data_stats/nn_parameters/total_passed_fp64_mb/agent_{self.agent_idx}": self.total_passed_fp64_mb,
                f"exchanged_data_stats/nn_parameters/total_exchanged_fp64_mb/agent_{self.agent_idx}": self.total_exchanged_fp64_mb,
                
                "global_communications": number_of_communications,
                "global_step": global_step,
            })
    
    def increase_exchanged_policy_table_parameters(self, number_of_communications, global_step):
        """
        Обновляет счетчики обмена параметрами таблицы политики и логирует информацию
        
        Args:
            number_of_communications: номер текущей глобальной коммуникации
            global_step: текущий глобальный шаг
        """
        args = self.args
        
        self.total_received_policy_table_parameters += args.policy_table_total_params_received_per_global_communication
        self.total_passed_policy_table_parameters += args.policy_table_total_params_passed_per_global_communication
        self.total_exchanged_policy_table_parameters = self.total_received_policy_table_parameters + self.total_passed_policy_table_parameters
        
        # Note: no logs to wandb, it is too large for current environments
        pass

    def log_memory_comparison(self, number_of_communications, global_step):
        """
        Логирует сравнительные метрики использования памяти между таблицей политики и нейронной сетью
        
        Args:
            number_of_communications: номер текущей глобальной коммуникации
            global_step: текущий глобальный шаг
        """
        pass
