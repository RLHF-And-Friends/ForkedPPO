#!/bin/bash

if [ -z "$1" ]; then
    echo "Ошибка: Не указан путь до файла commands.txt."
    echo "Использование: $0 <путь_до_commands.txt>"
    exit 1
fi

COMMANDS_FILE="$1"

export WANDB_API_KEY="88d8539f0a96d23135216aca56233e046cd229f6"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPUS=(1 2 4 5 6 7)
# CPU_SETS=("0-15" "16-31" "32-47")
CPU_SETS=("0-23" "24-47" "48-71")
# CPU_SETS=("0-11" "12-23" "24-35" "36-47")
# CPU_SETS=("0-7" "8-15" "16-23" "24-47")
MAX_JOBS=10

gpu_index=0
cpu_index=0

# Запускаем все команды из файла commands.txt
cat "$COMMANDS_FILE" | while read -r cmd; do
    if [ -z "$cmd" ]; then
        continue  # Пропускаем пустые строки
    fi
    
    # Извлекаем параметры из команды
    setup_id=$(echo "$cmd" | grep -o "\--setup-id=setup_[0-9]*" | sed 's/--setup-id=setup_//')
    seed=$(echo "$cmd" | grep -o "\--seed=[0-9]*" | sed 's/--seed=//')
    exp_name=$(echo "$cmd" | grep -o "\--exp-name=[^ ]*" | sed 's/--exp-name=//')
    env_type=$(echo "$cmd" | grep -o "\--env-type=[^ ]*" | sed 's/--env-type=//')

    if [ "$env_type" = "minigrid" ]; then
        logs_dir="federated_ppo/minigrid/logs"
    else
        # По умолчанию используем atari
        logs_dir="federated_ppo/atari/logs"
    fi

    # Создаем директорию для логов, если она не существует
    mkdir -p "$logs_dir"
    
    # Создаем уникальное имя файла для лога
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Формируем имя файла с доступными параметрами
    logfile="${logs_dir}/"
    if [ -n "$setup_id" ]; then
        logfile="${logfile}setup_${setup_id}_"
    fi
    if [ -n "$seed" ]; then
        logfile="${logfile}seed_${seed}_"
    fi
    if [ -n "$exp_name" ]; then
        logfile="${logfile}${exp_name}_"
    fi
    # Добавляем timestamp, чтобы избежать перезаписи при повторных запусках
    logfile="${logfile}${timestamp}.log"

    GPU="${GPUS[$gpu_index]}"
    CPU_CORES="${CPU_SETS[$cpu_index]}"
    echo "Выбран GPU = $GPU, CPU Cores = $CPU_CORES для команды: $cmd"

    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
    cpu_index=$(( (cpu_index + 1) % ${#CPU_SETS[@]} ))

    echo "Running: $cmd"  # Отладочный вывод команды
    echo "Log file: $logfile"  # Показываем, куда пишется лог

    CUDA_VISIBLE_DEVICES="$GPU" \
      taskset -c "$CPU_CORES" \
      $cmd > "$logfile" 2>&1 &
    # sg mygroup -c "$cmd > \"$logfile\" 2>&1 &"

    # Добавляем небольшую задержку между запусками
    sleep 1

    # Ограничиваем количество одновременно запущенных процессов
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 5
    done
done