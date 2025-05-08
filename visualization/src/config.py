import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get paths for visualization
BRIEFLOW_OUTPUT_PATH = os.environ["BRIEFLOW_OUTPUT_PATH"]
CONFIG_PATH = os.environ["CONFIG_PATH"]
SCREEN_PATH = os.environ["SCREEN_PATH"]

logger.info(f"CONFIG_PATH: {os.path.abspath(CONFIG_PATH)}")
logger.info(f"SCREEN_PATH: {os.path.abspath(SCREEN_PATH)}")


def load_config():
    """Load the YAML configuration file."""
    try:
        with open(CONFIG_PATH, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {os.path.abspath(CONFIG_PATH)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        raise
