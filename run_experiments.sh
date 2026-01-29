#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Path to commands.txt file not specified."
    echo "Usage: $0 <path_to_commands.txt>"
    exit 1
fi

COMMANDS_FILE="$1"
PARENT_DIR_NAME=$(basename "$(dirname "$COMMANDS_FILE")")

export WANDB_API_KEY="88d8539f0a96d23135216aca56233e046cd229f6"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPUS=(2)
# CPU_SETS=("0-15" "16-31" "32-47")
# CPU_SETS=("16-19" "20-23" "24-27" "28-31" "32-35")
CPU_SETS=("12-23")
# CPU_SETS=("0-11" "12-23" "24-35" "36-47")
# CPU_SETS=("0-7" "8-15" "16-23" "24-47")
MAX_JOBS=10

gpu_index=0
cpu_index=0

# Function to process multi-line commands
process_command() {
    local full_cmd="$1"

    # Remove backslashes and join lines
    full_cmd=$(echo "$full_cmd" | sed 's/\\\s*$//g' | tr -d '\n')

    # Extract parameters from the command
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

    # Create log directory if it does not exist
    mkdir -p "$logs_dir"

    # Write information about the current run to the history.log of the corresponding environment
    timestamp_now=$(date -d "+3 hours" +"%d/%m/%Y,%H:%M:%S")
    history_log="${env_logs_dir}/history.log"
    # If the file does not exist, create it with a header
    if [ ! -f "$history_log" ]; then
        echo "timestamp,commands_file" > "$history_log"
    fi
    # Add a new entry in CSV format
    echo "${timestamp_now},${COMMANDS_FILE}" >> "$history_log"

    # Create a unique log file name
    timestamp=$(date -d "+3 hours" +"%d_%m_%Y_%H_%M_%S")

    # Build the file name from available parameters
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
    # Add timestamp to avoid overwriting on repeated runs
    logfile="${logfile}${timestamp}.log"

    GPU="${GPUS[$gpu_index]}"
    CPU_CORES="${CPU_SETS[$cpu_index]}"
    echo "Selected GPU = $GPU, CPU Cores = $CPU_CORES for command: $full_cmd"

    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
    cpu_index=$(( (cpu_index + 1) % ${#CPU_SETS[@]} ))

    echo "Running: $full_cmd"  # Debug output of the command
    echo "Log file: $logfile"  # Show where the log is written

    CUDA_VISIBLE_DEVICES="$GPU" \
      taskset -c "$CPU_CORES" \
      $full_cmd > "$logfile" 2>&1 &
    # sg mygroup -c "$full_cmd > \"$logfile\" 2>&1 &"

    # Add a small delay between launches
    sleep 1

    # Limit the number of concurrently running processes
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 5
    done
}

# Read the file and assemble multi-line commands
current_command=""
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines if there is no current command
    if [ -z "$line" ] && [ -z "$current_command" ]; then
        continue
    fi

    # If the line is empty and we have a current command, it marks the end of the command
    if [ -z "$line" ] && [ -n "$current_command" ]; then
        process_command "$current_command"
        current_command=""
        continue
    fi

    # Append the line to the current command
    if [ -z "$current_command" ]; then
        current_command="$line"
    else
        current_command="$current_command
$line"
    fi
done < "$COMMANDS_FILE"

# Process the last command if it remains
if [ -n "$current_command" ]; then
    process_command "$current_command"
fi
