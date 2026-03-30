#!/usr/bin/env bash
#
# launch_unattended.sh
# Wisteria "Black Box" Protocol Launcher
#
set -e

# --- The Log File Contract ---
# This environment variable is the NON-NEGOTIABLE contract between this
# launcher and the main script's termination trap. It provides the
# single source of truth for the log file's location.
export WISTERIA_LOG_FILE="wisteria_run_$(date +"%Y%m%d-%H%M%S").log"

echo "========================================================================"
echo "WISTERIA UNATTENDED EXECUTION PROTOCOL ENGAGED"
echo "========================================================================"
echo "VM will automatically shut down on script completion or failure."
echo "Full log will be captured in: ${WISTERIA_LOG_FILE}"
echo "This log will be uploaded to GCS before shutdown."
echo ""
echo "Process is now detaching. You can safely close your SSH session."
echo "========================================================================"

# --- Process Detachment & Logging ---
# 'nohup' prevents the process from being terminated when the session ends.
# '>&' redirects both stdout and stderr to our single, canonical log file.
# '&' runs the process in the background.
nohup ./multi-headed-model_training_step2-trial20.sh > "${WISTERIA_LOG_FILE}" 2>&1 &