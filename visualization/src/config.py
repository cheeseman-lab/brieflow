import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get paths for visualization
BRIEFLOW_OUTPUT_PATH = os.environ["BRIEFLOW_OUTPUT_PATH"]
# CONFIG_PATH may list several YAML files, deep-merged left to right like `snakemake --configfile a b`
CONFIG_PATHS = os.environ["CONFIG_PATH"].split()
CONFIG_PATH = CONFIG_PATHS[0]
SCREEN_PATH = os.environ["SCREEN_PATH"]

# Static asset configuration - these can be None for local development
STATIC_ASSET_URL_ROOT = os.environ.get(
    "STATIC_ASSET_URL_ROOT", None
)  # e.g. "/aconcagua_dataset_static/"
STATIC_ASSET_PATH = os.environ.get(
    "STATIC_ASSET_PATH", None
)  # e.g. "/disk1/brieflow_datasets/aconcagua/"

logger.info(f"CONFIG_PATH: {[os.path.abspath(p) for p in CONFIG_PATHS]}")
logger.info(f"SCREEN_PATH: {os.path.abspath(SCREEN_PATH)}")


def load_config():
    """Load the YAML configuration, deep-merging every path in CONFIG_PATH."""
    config = {}
    for config_path in CONFIG_PATHS:
        try:
            with open(config_path, "r") as file:
                _deep_merge(config, yaml.safe_load(file) or {})
        except FileNotFoundError:
            logger.error(f"Config file not found at: {os.path.abspath(config_path)}")
            logger.info(f"Current working directory: {os.getcwd()}")
            raise
    return config


def get_image_format():
    """Return the pipeline output image format, either ``tiff`` or ``zarr``."""
    return load_config().get("all", {}).get("image_format", "tiff")


def _deep_merge(base, overlay):
    """Merge overlay into base in place, recursing into nested dicts."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
