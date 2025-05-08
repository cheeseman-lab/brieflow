import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the analysis directory from environment variable or use default
ANALYSIS_DIR = os.getenv('ANALYSIS_REPO_ROOT', '../../analysis/')
logger.info(f"ANALYSIS_DIR: {os.path.abspath(ANALYSIS_DIR)}")

# Derive paths from ANALYSIS_DIR
CONFIG_PATH = os.path.join(ANALYSIS_DIR, 'config', 'config.yml')
SCREEN_PATH = os.path.join(ANALYSIS_DIR, 'screen.yaml')

logger.info(f"CONFIG_PATH: {os.path.abspath(CONFIG_PATH)}")
logger.info(f"SCREEN_PATH: {os.path.abspath(SCREEN_PATH)}")

def load_config():
    """Load the YAML configuration file."""
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {os.path.abspath(CONFIG_PATH)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        raise

def get_analysis_root_dir():
    """Get the absolute path to the analysis root directory."""
    config = load_config()
    analysis_root = config['all']['root_fp']
    abs_path = os.path.abspath(
        os.path.join(
            os.path.dirname(CONFIG_PATH),
            '..',
            analysis_root))
    logger.info(f"Analysis root directory: {abs_path}")
    return abs_path 