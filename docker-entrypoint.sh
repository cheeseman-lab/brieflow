#!/usr/bin/env bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate brieflow

# Execute the command passed to the container
# If the command is passed as a single string, use bash -c
if [ $# -eq 1 ]; then
    exec bash -c "$1"
else
    exec "$@"
fi
