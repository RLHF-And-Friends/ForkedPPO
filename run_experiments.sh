#!/bin/bash

if [ -z "$1" ]; then
    echo "Ошибка: Не указан путь до файла commands.txt."
    echo "Использование: $0 <путь_до_commands.txt>"
    exit 1
fi

COMMANDS_FILE="$1"
PARENT_DIR_NAME=$(basename "$(dirname "$COMMANDS_FILE")")

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

# Функция для обработки многострочных команд
process_command() {
    local full_cmd="$1"
    
    # Удаляем обратные слеши и объединяем строки
    full_cmd=$(echo "$full_cmd" | sed 's/\\\s*$//g' | tr -d '\n')
    
    # Извлекаем параметры из команды
    setup_id=$(echo "$full_cmd" | grep -o "\--setup-id=[^ ]*" | sed 's/--setup-id=//')
    seed=$(echo "$full_cmd" | grep -o "\--seed=[0-9]*" | sed 's/--seed=//')
    exp_name=$(echo "$full_cmd" | grep -o "\--exp-name=[^ ]*" | sed 's/--exp-name=//')
    env_type=$(echo "$full_cmd" | grep -o "\--env-type=[^ ]*" | sed 's/--env-type=//')

    if [ "$env_type" = "minigrid" ]; then
        logs_dir="federated_ppo/minigrid/logs/${PARENT_DIR_NAME}"
        env_logs_dir="federated_ppo/minigrid/logs"
    elif [ "$env_type" = "mujoco" ]; then
        logs_dir="federated_ppo/mujoco/logs/${PARENT_DIR_NAME}"
        env_logs_dir="federated_ppo/mujoco/logs"
    else
        logs_dir="federated_ppo/atari/logs/${PARENT_DIR_NAME}"
        env_logs_dir="federated_ppo/atari/logs"
    fi

    # Создаем директорию для логов, если она не существует
    mkdir -p "$logs_dir"
    
    # Записываем информацию о текущем запуске в history.log соответствующего окружения
    timestamp_now=$(date -d "+3 hours" +"%d/%m/%Y,%H:%M:%S")
    history_log="${env_logs_dir}/history.log"
    # Если файл не существует, создаем его с заголовком
    if [ ! -f "$history_log" ]; then
        echo "timestamp,commands_file" > "$history_log"
    fi
    # Добавляем новую запись в формате CSV
    echo "${timestamp_now},${COMMANDS_FILE}" >> "$history_log"
    
    # Создаем уникальное имя файла для лога
    timestamp=$(date -d "+3 hours" +"%d_%m_%Y_%H_%M_%S")
    
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
    echo "Выбран GPU = $GPU, CPU Cores = $CPU_CORES для команды: $full_cmd"

    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
    cpu_index=$(( (cpu_index + 1) % ${#CPU_SETS[@]} ))

    echo "Running: $full_cmd"  # Отладочный вывод команды
    echo "Log file: $logfile"  # Показываем, куда пишется лог

    CUDA_VISIBLE_DEVICES="$GPU" \
      taskset -c "$CPU_CORES" \
      $full_cmd > "$logfile" 2>&1 &
    # sg mygroup -c "$full_cmd > \"$logfile\" 2>&1 &"

    # Добавляем небольшую задержку между запусками
    sleep 1

    # Ограничиваем количество одновременно запущенных процессов
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 5
    done
}

# Читаем файл и собираем многострочные команды
current_command=""
while IFS= read -r line || [ -n "$line" ]; do
    # Пропускаем пустые строки, если нет текущей команды
    if [ -z "$line" ] && [ -z "$current_command" ]; then
        continue
    fi
    
    # Если строка пустая и у нас есть текущая команда, это конец команды
    if [ -z "$line" ] && [ -n "$current_command" ]; then
        process_command "$current_command"
        current_command=""
        continue
    fi
    
    # Добавляем строку к текущей команде
    if [ -z "$current_command" ]; then
        current_command="$line"
    else
        current_command="$current_command
$line"
    fi
done < "$COMMANDS_FILE"

# Обрабатываем последнюю команду, если она осталась
if [ -n "$current_command" ]; then
    process_command "$current_command"
fi