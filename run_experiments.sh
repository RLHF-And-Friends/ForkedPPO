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
CPU_SETS=("0-7" "8-15" "16-23" "24-31" "32-39" "40-47" "48-55" "56-63")
# CPU_SETS=("0-11" "12-23" "24-35" "36-47")
# CPU_SETS=("0-7" "8-15" "16-23" "24-47")
MAX_JOBS=10

mkdir -p logs

gpu_index=0
cpu_index=0

# Запускаем все команды из файла commands.txt
cat "$COMMANDS_FILE" | while read -r cmd; do
    if [ -z "$cmd" ]; then
        continue  # Пропускаем пустые строки
    fi
    
    # setup_id=$(echo "$cmd" | awk -F'--setup-id=setup_' '{print $2}' | awk '{print $1}')

    # if [ -z "$setup_id" ]; then
    #     echo "Error: Could not extract setup-id from command: $cmd"
    #     continue
    # fi

    GPU="${GPUS[$gpu_index]}"
    CPU_CORES="${CPU_SETS[$cpu_index]}"
    echo "Выбран GPU = $GPU, CPU Cores = $CPU_CORES для команды: $cmd"

    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
    cpu_index=$(( (cpu_index + 1) % ${#CPU_SETS[@]} ))

    echo "Running: $cmd"  # Отладочный вывод команды
    # logfile="logs/setup_${setup_id}.log"
    logfile="logs/tmp.log"
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