#!/usr/bin/env bash

# Default port and config
#export STREAMLIT_CONFIG=".streamlit/config.toml"

cd /home/andy/work/src/brieflow-analysis/brieflow/visualization

# Set default analysis directory if not provided
# Override this by setting ANALYSIS_DIR before running the script
if [ -z "$ANALYSIS_DIR" ]; then
    export ANALYSIS_DIR="../../analysis/"
fi

# Start Streamlit server, force bind to 0.0.0.0
exec streamlit run Experimental_Overview.py --server.address=0.0.0.0 "$@"
