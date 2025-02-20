import json
import logging
import os

def load_config(config_path='config.json'):
    """
    Loads configuration from a JSON file. If the file is not found or JSON is invalid,
    returns a default configuration and logs a warning.
    """

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration loaded from '{config_path}'")
            return config

    except Exception as e:
        logging.error(f"Unexpected error loading configuration from '{config_path}': {e}. Using default configuration.")

