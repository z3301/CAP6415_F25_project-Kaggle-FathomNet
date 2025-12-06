#!/bin/bash
# Training Monitor Script for FathomNet Multi-Scale Experiments
#
# This script monitors the training progress of Exp 1a and 1b and alerts when complete.
#
# Usage:
#   bash monitor_training.sh

OUTPUT_DIR="/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/outputs"
EXP1A_DIR="${OUTPUT_DIR}/exp1a_2scales"
EXP1B_DIR="${OUTPUT_DIR}/exp1b_3scales"
LOG_FILE="${OUTPUT_DIR}/training_monitor.log"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Initialize log file
echo "=== FathomNet Training Monitor Started: $(date) ===" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Function to check if experiment has completed
check_experiment_status() {
    local exp_name=$1
    local exp_dir=$2

    # Check if lightning_logs directory exists
    if [ ! -d "${exp_dir}/lightning_logs" ]; then
        echo "⏳ ${exp_name}: Not started yet (no lightning_logs found)" | tee -a "${LOG_FILE}"
        return 1
    fi

    # Check if any checkpoint files exist
    checkpoint_count=$(find "${exp_dir}" -name "*.ckpt" 2>/dev/null | wc -l)

    if [ ${checkpoint_count} -eq 0 ]; then
        echo "⏳ ${exp_name}: Training in progress (no checkpoints yet)" | tee -a "${LOG_FILE}"
        return 1
    fi

    # Check if training is still active by looking for recent file modifications
    # Consider training complete if no files modified in last 30 minutes
    recent_files=$(find "${exp_dir}/lightning_logs" -type f -mmin -30 2>/dev/null | wc -l)

    if [ ${recent_files} -eq 0 ]; then
        echo "✅ ${exp_name}: COMPLETE (${checkpoint_count} checkpoints saved)" | tee -a "${LOG_FILE}"
        return 0
    else
        # Get latest epoch from version_0 if it exists
        if [ -d "${exp_dir}/lightning_logs/version_0" ]; then
            latest_epoch=$(find "${exp_dir}/lightning_logs/version_0" -name "events.out.tfevents.*" -exec ls -lt {} + 2>/dev/null | head -1 | awk '{print $9}')
            echo "⏳ ${exp_name}: Training in progress (${checkpoint_count} checkpoints, active)" | tee -a "${LOG_FILE}"
        else
            echo "⏳ ${exp_name}: Training in progress" | tee -a "${LOG_FILE}"
        fi
        return 1
    fi
}

# Function to get training summary
get_training_summary() {
    local exp_name=$1
    local exp_dir=$2

    echo "" | tee -a "${LOG_FILE}"
    echo "--- ${exp_name} Summary ---" | tee -a "${LOG_FILE}"

    # Find best checkpoint (lowest val_loss)
    best_ckpt=$(find "${exp_dir}" -name "*.ckpt" -type f 2>/dev/null | sort | head -1)

    if [ -n "${best_ckpt}" ]; then
        echo "Best checkpoint: ${best_ckpt}" | tee -a "${LOG_FILE}"

        # Extract epoch and val_loss from filename if possible
        if [[ ${best_ckpt} =~ epoch=([0-9]+).*val_loss=([0-9.]+) ]]; then
            epoch="${BASH_REMATCH[1]}"
            val_loss="${BASH_REMATCH[2]}"
            echo "  Epoch: ${epoch}" | tee -a "${LOG_FILE}"
            echo "  Val Loss: ${val_loss}" | tee -a "${LOG_FILE}"
        fi
    fi

    # Count total checkpoints
    checkpoint_count=$(find "${exp_dir}" -name "*.ckpt" 2>/dev/null | wc -l)
    echo "  Total checkpoints: ${checkpoint_count}" | tee -a "${LOG_FILE}"

    echo "" | tee -a "${LOG_FILE}"
}

# Main monitoring loop
echo "Monitoring experiments..." | tee -a "${LOG_FILE}"
echo "  - Exp 1a: ${EXP1A_DIR}" | tee -a "${LOG_FILE}"
echo "  - Exp 1b: ${EXP1B_DIR}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

exp1a_done=false
exp1b_done=false
check_count=0

while true; do
    check_count=$((check_count + 1))
    echo "=== Check #${check_count}: $(date) ===" | tee -a "${LOG_FILE}"

    # Check Exp 1a
    if [ "${exp1a_done}" = false ]; then
        if check_experiment_status "Exp 1a" "${EXP1A_DIR}"; then
            exp1a_done=true
            get_training_summary "Exp 1a" "${EXP1A_DIR}"

            # Send desktop notification if available
            if command -v notify-send &> /dev/null; then
                notify-send "FathomNet Training" "Exp 1a (2 scales) has completed!"
            fi
        fi
    fi

    # Check Exp 1b
    if [ "${exp1b_done}" = false ]; then
        if check_experiment_status "Exp 1b" "${EXP1B_DIR}"; then
            exp1b_done=true
            get_training_summary "Exp 1b" "${EXP1B_DIR}"

            # Send desktop notification if available
            if command -v notify-send &> /dev/null; then
                notify-send "FathomNet Training" "Exp 1b (3 scales) has completed!"
            fi
        fi
    fi

    echo "" | tee -a "${LOG_FILE}"

    # Exit if both experiments are done
    if [ "${exp1a_done}" = true ] && [ "${exp1b_done}" = true ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
        echo "🎉 ALL EXPERIMENTS COMPLETE! 🎉" | tee -a "${LOG_FILE}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "Next steps:" | tee -a "${LOG_FILE}"
        echo "  1. Run evaluation for Exp 1a:" | tee -a "${LOG_FILE}"
        echo "     python evaluate_multiscale.py --model ${EXP1A_DIR}/<best_checkpoint>.ckpt" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "  2. Run evaluation for Exp 1b:" | tee -a "${LOG_FILE}"
        echo "     python evaluate_multiscale.py --model ${EXP1B_DIR}/<best_checkpoint>.ckpt" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "  3. Compare results and proceed to Day 3 of timeline" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"

        # Send final notification
        if command -v notify-send &> /dev/null; then
            notify-send "FathomNet Training" "🎉 ALL EXPERIMENTS COMPLETE! Ready for evaluation."
        fi

        break
    fi

    # Wait 30 minutes before next check (adjust as needed)
    echo "Sleeping 30 minutes until next check..." | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    sleep 1800  # 30 minutes
done

echo "Monitor script finished: $(date)" | tee -a "${LOG_FILE}"
