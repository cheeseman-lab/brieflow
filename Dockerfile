FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /opt/brieflow

# Copy the entire Brieflow codebase
COPY . /opt/brieflow

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Accept conda Terms of Service and create environment with uv
# Using Python 3.11 for better package compatibility
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --set auto_activate_base false && \
    echo "yes" | conda create -n brieflow -c conda-forge python=3.11 uv pip -y && \
    echo "source activate brieflow" > ~/.bashrc

# Activate environment and install dependencies
SHELL ["conda", "run", "-n", "brieflow", "/bin/bash", "-c"]

# Copy pyproject.toml for dependency installation
COPY pyproject.toml /opt/brieflow/pyproject.toml

# Install dependencies using uv (fast!)
RUN uv pip install -r pyproject.toml

# Install brieflow package in editable mode
RUN uv pip install -e .

# Make sure the conda environment is activated by default
ENV PATH /opt/conda/envs/brieflow/bin:$PATH
ENV CONDA_DEFAULT_ENV brieflow
# Ensure Python can find all installed packages
ENV PYTHONPATH /opt/conda/envs/brieflow/lib/python3.11/site-packages:$PYTHONPATH

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

# Set the working directory to where Snakemake will run
WORKDIR /workdir

# Use entrypoint that activates conda environment
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/bin/bash"]
