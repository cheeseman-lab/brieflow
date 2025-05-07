#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"

# Set default analysis directory if not provided
# Override this by setting ANALYSIS_DIR before running the script
if [ -z "$ANALYSIS_DIR" ]; then
    export ANALYSIS_DIR="../../analysis/"
fi

# Start Streamlit server, force bind to 0.0.0.0
exec streamlit run Experimental_Overview.py --server.address=0.0.0.0 "$@"