# Brieflow Visualization

This directory contains the visualization tools for analyzing Brieflow screen data.

## Running the Visualization

The visualization can be run using the `run-visualization.sh` script. The script requires an analysis directory path to be specified, which can be done in two ways:

1. Using the default path (relative to the visualization directory):
```bash
./run-visualization.sh
```

2. Specifying a custom analysis directory path:
```bash
ANALYSIS_DIR=/path/to/analysis/dir ./run-visualization.sh
```

## Expected Directory Structure

The `ANALYSIS_DIR` should point to a directory that follows this structure:

```
analysis/
├── brieflow_output/     # Contains all outputs including TSV files and images
├── screen.yml        # Screen configuration file
└── config/
    └── config.yml    # Analysis configuration file
```

For example, if your analysis is located at `/lab/cheeseman_ops/brieflow-screens/denali-analysis/analysis`, you would run:

```bash
ANALYSIS_DIR=/lab/cheeseman_ops/brieflow-screens/denali-analysis/analysis ./run-visualization.sh
```

## Environment Variables

- `ANALYSIS_DIR`: The root directory of your analysis. This is the only required environment variable. If not set, it defaults to `../analysis/` relative to the visualization directory.

## Visualization Features

The visualization tool provides an interactive interface for exploring your analysis results. It automatically loads:
- Screen configuration from `screen.yml`
- Analysis configuration from `config/config.yml`
- Analysis outputs from the `brieflow_output/` directory 