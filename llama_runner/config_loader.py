import os
import json
import logging

CONFIG_DIR = os.path.expanduser("~/.llama-runner")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
LOG_FILE = os.path.join(CONFIG_DIR, "error.log")

# Ensure the log directory exists
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_config_exists():
    """
    Ensures that the configuration directory and file exist.
    Creates them if they don't.
    """
    if not os.path.exists(CONFIG_DIR):
        try:
            os.makedirs(CONFIG_DIR)
        except OSError as e:
            print(f"Error creating config directory: {e}")
            logging.error(f"Error creating config directory: {e}")
            return False

    if not os.path.exists(CONFIG_FILE):
        try:
            default_config = {
                "models": {},
                "llama-runtimes": {},
                "default_runtime": "llama-server",
                "concurrentRunners": 1,
                "proxies": {
                    "ollama": {"enabled": True},
                    "lmstudio": {"enabled": True, "api_key": None}
                },
                "logging": {"prompt_logging_enabled": False}
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(default_config, f, indent=2)
        except OSError as e:
            print(f"Error creating config file: {e}")
            logging.error(f"Error creating config file: {e}")
            return False
    return True

def load_config():
    """
    Loads the configuration from the JSON file.
    Returns a dictionary containing the configuration.
    """
    if not ensure_config_exists():
        return {}

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

            # Ensure default_runtime and concurrentRunners exist
            if "default_runtime" not in config:
                config["default_runtime"] = "llama-server"
            if "concurrentRunners" not in config:
                config["concurrentRunners"] = 1

            # Ensure proxies section and its sub-keys exist with defaults
            proxies_config = config.get("proxies", {})
            if not isinstance(proxies_config, dict): # Handle case where 'proxies' might exist but not as a dict
                proxies_config = {}

            ollama_proxy_config = proxies_config.get("ollama", {})
            if not isinstance(ollama_proxy_config, dict):
                ollama_proxy_config = {}
            if "enabled" not in ollama_proxy_config:
                ollama_proxy_config["enabled"] = True
            proxies_config["ollama"] = ollama_proxy_config

            lmstudio_proxy_config = proxies_config.get("lmstudio", {})
            if not isinstance(lmstudio_proxy_config, dict):
                lmstudio_proxy_config = {}
            if "enabled" not in lmstudio_proxy_config:
                lmstudio_proxy_config["enabled"] = True
            if "api_key" not in lmstudio_proxy_config: # lmstudio might not always have api_key, default to None
                lmstudio_proxy_config["api_key"] = None
            proxies_config["lmstudio"] = lmstudio_proxy_config
            
            config["proxies"] = proxies_config

            # Ensure logging section and its sub-keys exist with defaults
            logging_config = config.get("logging", {})
            if not isinstance(logging_config, dict): # Handle case where 'logging' might exist but not as a dict
                logging_config = {}
            if "prompt_logging_enabled" not in logging_config:
                logging_config["prompt_logging_enabled"] = False
            config["logging"] = logging_config
            
            # Process llama-runtimes to normalize structure
            raw_runtimes = config.get("llama-runtimes")
            if isinstance(raw_runtimes, dict):
                processed_runtimes = {}
                for name, details in raw_runtimes.items():
                    if isinstance(details, str):  # Old format: "runtime_name": "command"
                        if details.strip(): # Ensure command is not empty
                            processed_runtimes[name] = {
                                "runtime": details,
                                "supports_tools": True  # Default for old format
                            }
                        else:
                            logging.warning(f"Config: Runtime entry '{name}' (old format) has an empty command. Skipping.")
                            print(f"Warning: Config: Runtime entry '{name}' (old format) has an empty command. Skipping.")
                    elif isinstance(details, dict): # New format: "runtime_name": {"runtime": "command", "supports_tools": False/True}
                        if "runtime" in details:
                            runtime_cmd = details["runtime"]
                            if isinstance(runtime_cmd, str) and runtime_cmd.strip(): # Check if runtime command is a non-empty string
                                processed_runtimes[name] = {
                                    "runtime": runtime_cmd,
                                    "supports_tools": details.get("supports_tools", True)
                                }
                            else: # Invalid or empty runtime command
                                logging.warning(f"Config: Runtime entry '{name}' has an invalid or empty 'runtime' command value. Skipping.")
                                print(f"Warning: Config: Runtime entry '{name}' has an invalid or empty 'runtime' command value. Skipping.")
                        else: # 'runtime' key is missing
                            logging.warning(f"Config: Runtime entry '{name}' in 'llama-runtimes' is missing 'runtime' key. Skipping.")
                            print(f"Warning: Config: Runtime entry '{name}' in 'llama-runtimes' is missing 'runtime' key. Skipping.")
                    else: # Invalid type for runtime details
                        logging.warning(f"Config: Runtime entry '{name}' in 'llama-runtimes' has invalid format (expected string or dict). Skipping.")
                        print(f"Warning: Config: Runtime entry '{name}' in 'llama-runtimes' has invalid format. Skipping.")
                config["llama-runtimes"] = processed_runtimes # Update config with processed runtimes
            elif raw_runtimes is not None: # 'llama-runtimes' exists but is not a dictionary
                logging.warning("Config: 'llama-runtimes' key exists but is not a dictionary. Ignoring.")
                print("Warning: Config: 'llama-runtimes' key exists but is not a dictionary. Ignoring.")
            # If 'llama-runtimes' is not in config or is None, it's handled gracefully (no changes made to it)

            print(f"Loaded config (processed): {config}")  # Print loaded config
            return config
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        logging.error(f"Error loading config file: {e}")
        return {}

def calculate_system_fingerprint(config: dict) -> str:
    """Calculates a 16-character hash of the configuration parameters."""
    import hashlib
    import json
    config_str = json.dumps(config, sort_keys=True)
    hash_object = hashlib.md5(config_str.encode())
    return hash_object.hexdigest()[:16]

if __name__ == '__main__':
    # Example usage:
    config = load_config()
    print(f"Loaded config: {config}")
