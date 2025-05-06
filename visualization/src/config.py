import os
import yaml

# Default path to the configuration file
CONFIG_PATH = os.getenv('BRIEFLOW_CONFIG_PATH', '../../analysis/config/config.yml')

# Default path to the screen configuration file
SCREEN_PATH = os.getenv('BRIEFLOW_SCREEN_PATH', '../../screen.yaml')

def load_config():
    """Load the YAML configuration file."""
    with open(CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

def get_analysis_root_dir():
    """Get the absolute path to the analysis root directory."""
    config = load_config()
    analysis_root = config['all']['root_fp']
    return os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(CONFIG_PATH)), 
                '..', 
                'analysis', 
                analysis_root)) 