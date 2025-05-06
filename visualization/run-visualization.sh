#!/usr/bin/env bash

# Default port and config
#export STREAMLIT_CONFIG=".streamlit/config.toml"

cd /home/andy/work/src/brieflow-analysis/brieflow/visualization

if [ -z "$SCREEN_OVERVIEW_PATH" ]; then
    export BRIEFLOW_SCREEN_PATH="../../screen.yaml"
fi

if [ -z "$CONFIG_PATH" ]; then
    export BRIEFLOW_CONFIG_PATH="../../analysis/config/config.yml"
fi

# Start Streamlit server, force bind to 0.0.0.0
exec streamlit run Screen_Overview.py --server.address=0.0.0.0 "$@"
