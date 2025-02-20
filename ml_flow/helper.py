import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path=r'../ml_flow/config.json'):
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
        else:
            logging.warning(f"Configuration file '{config_path}' not found. Using default settings.")

    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in '{config_path}'. Using default settings.")

    except Exception as e:
        logging.error(f"Unexpected error loading configuration from '{config_path}': {e}.")

    return {}  # Return a default empty config instead of None

def main():
    config = load_config()
    print("Loaded Configuration:", config)

if __name__ == '__main__':
    main()
