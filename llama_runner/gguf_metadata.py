import os
import json
import logging
import traceback # Import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from llama_runner.config_loader import CONFIG_DIR # Assuming CONFIG_DIR is defined here

# Attempt to import numpy
np = None # Initialize to None for Pylance
try:
    import numpy as np_imported # Import with a temporary name
    np = np_imported # Assign to the desired name 'np' if import succeeds
    NUMPY_AVAILABLE = True
except ImportError:
    # np remains None as initialized above
    logging.warning("The 'numpy' library is not installed. Handling of numpy array metadata may be limited.")
    NUMPY_AVAILABLE = False

# Attempt to import the gguf library and specific components
GGUFReader = None # Initialize to None for Pylance
LlamaFileType = None # Initialize to None for Pylance
try:
    from gguf import GGUFReader as GGUFReader_imported # Import with a temporary name
    from gguf.constants import LlamaFileType as LlamaFileType_imported # Import with a temporary name
    GGUFReader = GGUFReader_imported # Assign if import succeeds
    LlamaFileType = LlamaFileType_imported # Assign if import succeeds
    GGUF_AVAILABLE = True
    logging.debug("Successfully imported GGUFReader and LlamaFileType.")
except ImportError as e:
    # GGUFReader and LlamaFileType remain None as initialized above
    logging.warning(f"ImportError: The 'gguf' library or required components are missing: {e}. Metadata extraction will be disabled.")
    GGUF_AVAILABLE = False
except Exception as e:
    # GGUFReader and LlamaFileType remain None as initialized above
    logging.warning(f"Error importing gguf components: {e}. Metadata extraction may be limited.")
    GGUF_AVAILABLE = False

# --- Add debug logging for GGUF_AVAILABLE status ---
logging.debug(f"GGUF_AVAILABLE status after import attempt: {GGUF_AVAILABLE}")
# --- End debug logging ---


METADATA_CACHE_DIR = os.path.join(CONFIG_DIR, "metadata_cache")
Path(METADATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

def ensure_cache_dir_exists():
    """Ensures the metadata cache directory exists."""
    Path(METADATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured metadata cache directory exists: {METADATA_CACHE_DIR}")

# Renamed from calculate_file_hash to get_file_size
def get_file_size(filepath: str) -> Optional[int]:
    """Gets the size of a file in bytes."""
    try:
        return os.path.getsize(filepath)
    except FileNotFoundError:
        logging.error(f"File not found for size check: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error getting size for {filepath}: {e}\n{traceback.format_exc()}")
        return None

# Updated to use file_size instead of file_hash
def get_metadata_cache_path(model_name: str, file_size: int) -> str:
    """Generates a cache file path based on model name and file size."""
    # Sanitize model name for filename
    safe_model_name = "".join(c if c.isalnum() or c in (' ', '.', '-') else '_' for c in model_name).replace(' ', '_')
    # Use file size in the filename
    return os.path.join(METADATA_CACHE_DIR, f"{safe_model_name}_{file_size}.json")

# Updated to use file_size instead of file_hash
def load_metadata_from_cache(model_name: str, file_size: int) -> Optional[Dict[str, Any]]:
    """Loads metadata from the cache file if it exists and is valid."""
    cache_path = get_metadata_cache_path(model_name, file_size)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from cache file {cache_path}: {e}")
            # In case of error, treat cache as invalid
            return None
        except Exception as e:
            logging.error(f"Error loading metadata from cache {cache_path}: {e}")
            return None
    return None

# Helper function to prepare data for JSON serialization
def prepare_for_json(data: Any) -> Any:
    """
    Recursively prepares data for JSON serialization by converting
    unsupported types like numpy arrays/memmaps.
    """
    if isinstance(data, dict):
        return {k: prepare_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [prepare_for_json(item) for item in data]
    # Ensure np is not None before using its attributes
    elif NUMPY_AVAILABLE and np is not None and isinstance(data, (np.ndarray, np.memmap)): # Explicit np is not None
        if data.size == 1:
            # Convert singleton numpy arrays to their scalar item
            try:
                return data.item()
            except Exception:
                # Fallback to string representation if item() fails
                return str(data)
        elif data.ndim == 1:
            # Convert 1D arrays to string representation
            try:
                if data.dtype == np.uint8 or data.dtype == np.int8:
                    # Assume bytes and decode
                    return bytes(data).decode('utf-8', errors='replace')
                else:
                    # Convert numeric array to string
                    # Use a very large int for max_line_width instead of np.inf
                    return np.array2string(data, separator=', ', max_line_width=1000000)
            except Exception:
                # Fallback to generic string representation
                return str(data)
        else:
            # Convert other numpy arrays to string representation
            return str(data)
    else:
        # Return the data as is if it's already JSON serializable
        return data

# Updated to use file_size instead of file_hash
def save_metadata_to_cache(model_name: str, file_size: int, metadata: Dict[str, Any]):
    """Saves metadata to a cache file."""
    cache_path = get_metadata_cache_path(model_name, file_size)
    try:
        # Prepare metadata for JSON serialization
        prepared_metadata = prepare_for_json(metadata)
        with open(cache_path, 'w') as f:
            json.dump(prepared_metadata, f, indent=2)
        logging.info(f"Saved metadata to cache for {model_name} (size: {file_size}) at {cache_path}")
    except Exception as e:
        logging.error(f"Error saving metadata to cache {cache_path}: {e}")

def extract_gguf_metadata(model_path: str, model_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts relevant metadata from a GGUF file."""
    # --- Add debug logging at the start of the function ---
    logging.debug(f"Attempting to extract GGUF metadata from: {model_path}")
    # --- End debug logging ---

    if not GGUF_AVAILABLE:
        logging.error("GGUF library not available. Cannot extract metadata.")
        return None
    if not os.path.exists(model_path):
        logging.error(f"Model file not found for metadata extraction: {model_path}")
        return None

    if not GGUFReader: # Check if GGUFReader was successfully imported (it's initialized to None if import fails)
        logging.error("GGUFReader not available due to import error. Cannot extract metadata.")
        return None

    try:
        reader = GGUFReader(model_path, 'r')
        raw_metadata = {} # Initialize raw_metadata dictionary
        final_metadata = {} # Initialize the dictionary to be returned

        # Iterate through all fields and extract key-value pairs
        for key, field in reader.fields.items():
            try:
                # Access the value using the method shown in the example reader.py
                # field.data[0] is the index into field.parts
                # This might return a list/tuple or array-like object even for scalar types
                value = field.parts[field.data[0]]
                raw_metadata[key] = value # Store in raw_metadata
                # Add detailed logging for extracted values, especially non-scalar ones
                if isinstance(value, (list, tuple)):
                     logging.debug(f"Extracted list/tuple metadata for key '{key}': Type={type(value)}, Length={len(value)}, Value={value}")
                # Ensure np is not None before using its attributes
                elif NUMPY_AVAILABLE and np is not None and isinstance(value, (np.ndarray, np.memmap)): # Explicit np is not None
                     logging.debug(f"Extracted numpy array metadata for key '{key}': Type={type(value)}, Shape={value.shape}, Value={value}")
                else:
                     logging.debug(f"Extracted scalar metadata for key '{key}': Type={type(value)}, Value={value}")

            except (IndexError, TypeError, AttributeError) as e:
                logging.warning(f"Could not extract value for key '{key}' from {model_path}: {e}")
                raw_metadata[key] = None # Store None or skip? Store None for now.
            except Exception as e:
                 # Catch any other unexpected errors during extraction
                 logging.warning(f"Unexpected error extracting value for key '{key}' from {model_path}: {e}\n{traceback.format_exc()}")
                 raw_metadata[key] = None

        # Helper to safely get a scalar value from a metadata dictionary, handling lists/tuples/arrays
        # Returns the raw scalar value, does not force string conversion
        def get_scalar_metadata(metadata_dict: Dict[str, Any], key: str, default: Any = None) -> Any:
            value = metadata_dict.get(key)

            # Handle numpy arrays/memmaps first
            # Ensure np is not None before using its attributes
            if NUMPY_AVAILABLE and np is not None and isinstance(value, (np.ndarray, np.memmap)): # Explicit np is not None
                if value.size == 1:
                    try:
                        return value.item() # Extracts the single scalar value
                    except Exception as e:
                        logging.warning(f"Could not extract scalar item from numpy array for key '{key}': {e}")
                        return default
                elif value.ndim == 1: # 1D array
                    try:
                        if value.dtype == np.uint8 or value.dtype == np.int8: # Byte array
                            return bytes(value).decode('utf-8', errors='replace')
                        else: # Other numeric 1D arrays
                            # Return as a list of Python native types if possible, or string as fallback
                            return value.tolist() # Converts to Python list
                    except Exception as e:
                        logging.warning(f"Could not convert 1D numpy array for key '{key}': {e}")
                        return str(value) # Fallback to string representation
                else: # Multi-dimensional arrays
                    logging.warning(f"Unsupported numpy array shape/ndim for key '{key}': {value.shape}. Returning as string.")
                    return str(value) # Fallback to string representation

            # Keep unwrapping lists/tuples until a non-container or None is found
            while isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]

            # If value is a string that represents a number, try to convert it
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        pass # Not an int or float string, return as is

            if value is None:
                 return default
            else:
                # Return the raw value if it's not a list/tuple/array (after unwrapping)
                return value


        # --- Construct LM Studio format based on requested fields, using raw_metadata ---

        # id: general.name (fallback to filename) or use model_config["model_id"] if specified
        if "model_id" in model_config:
            model_id = model_config["model_id"]
        else:
            model_id_val = get_scalar_metadata(raw_metadata, 'general.name', os.path.basename(model_path))
            model_id = str(model_id_val) if model_id_val is not None else os.path.basename(model_path)

        # object: "model" (static)
        obj_type = "model" # Static value as per LM Studio API

        # type: "llm", "vlm", "embeddings" (detected)
        # Check metadata keys first
        model_type = "llm" # Default to llm
        ggml_model_type_val = get_scalar_metadata(raw_metadata, "ggml.model.type")
        if ggml_model_type_val is not None and isinstance(ggml_model_type_val, str) and ggml_model_type_val.lower() == "embedding":
             model_type = "embeddings"
        elif ggml_model_type_val is not None and isinstance(ggml_model_type_val, str) and ggml_model_type_val.lower() == "vlm":
             model_type = "vlm"
        # Fallback to filename heuristic if metadata key is missing
        elif "embedding" in os.path.basename(model_path).lower() or "embed" in os.path.basename(model_path).lower():
             model_type = "embeddings"
        # VLM detection is harder, might need specific metadata keys or filename patterns
        # For now, stick to llm/embeddings based on metadata/filename heuristic


        # publisher: general.quantized_by (fallback to general.url, then local)
        publisher_val = get_scalar_metadata(raw_metadata, 'general.quantized_by', get_scalar_metadata(raw_metadata, 'general.url', 'local'))
        # Ensure publisher is a string
        publisher = str(publisher_val) if publisher_val is not None else 'local'

        architecture_val = get_scalar_metadata(raw_metadata, 'general.architecture', 'unknown')
        # Ensure architecture is a string
        architecture = str(architecture_val) if architecture_val is not None else 'unknown'

        # compatibility_type: "gguf" (static)
        compatibility_type = "gguf" # Static value

        # quantization: LlamaFileType(general.file_type).name (fallback to heuristic)
        quantization = "Unknown"
        file_type_val = get_scalar_metadata(raw_metadata, 'general.file_type') # Use the helper to get the raw value

        # --- Add debug logging for file_type_val ---
        logging.debug(f"Raw 'general.file_type' value: {file_type_val}, Type: {type(file_type_val)}")
        # --- End debug logging ---
        if GGUF_AVAILABLE and file_type_val is not None:
            try:
                # Attempt to convert file_type_val to an integer
                if isinstance(file_type_val, str): # If it's a string, try to convert to int
                    try:
                        file_type_int = int(file_type_val)
                    except ValueError:
                        logging.warning(f"Could not convert 'general.file_type' string '{file_type_val}' to int for {model_path}.")
                        file_type_int = None
                elif isinstance(file_type_val, int): # If it's already an int
                    file_type_int = file_type_val
                else: # If it's neither string nor int
                    logging.warning(f"'general.file_type' is not an integer or string ({type(file_type_val)}) in {model_path}. Value: {file_type_val}")
                    file_type_int = None # Fall through to heuristic

                if file_type_int is not None and LlamaFileType is not None: # Check LlamaFileType availability (it's initialized to None if import fails)
                    try:
                        # Use the integer value to get the enum name from LlamaFileType
                        quantization = LlamaFileType(file_type_int).name
                    except ValueError: # Handles cases where the int value is not a valid member of LlamaFileType
                        logging.warning(f"Integer value {file_type_int} for 'general.file_type' is not a valid LlamaFileType member for {model_path}.")
                        quantization = f"Type_{file_type_int}" # Fallback if enum value is unknown
                # else: file_type_int is None, fall through to heuristic

            except Exception as e: # Catch other exceptions during LlamaFileType processing
                 logging.warning(f"Error processing 'general.file_type' ({file_type_val}) for {model_path}: {e}")
                 # Fall through to heuristic
        # Fallback to heuristic if enum method fails or is not available
        if quantization == "Unknown":
             if "q4_k_m" in os.path.basename(model_path).lower():
                  quantization = "Q4_K_M" # Common convention
             # Fallback: Check if quantization info is directly in metadata keys (e.g., "quantization.method")
             elif get_scalar_metadata(raw_metadata, "quantization.method"):
                  quantization = get_scalar_metadata(raw_metadata, "quantization.method")
             elif get_scalar_metadata(raw_metadata, "quantization_version"): # Sometimes just a version number
                  quantization = f"Q{get_scalar_metadata(raw_metadata, 'quantization_version')}"

        # --- Remove "MOSTLY_" prefix if it exists ---
        if isinstance(quantization, str) and quantization.startswith("MOSTLY_"):
             quantization = quantization[len("MOSTLY_"):]
        # --- End prefix removal ---

        # Ensure quantization is a string
        quantization = str(quantization) if quantization is not None else 'Unknown'


        # max_context_length: ${general.architecture}.context_length (fallback to default)
        max_ctx = 4096 # Default value
        # Use the architecture string obtained earlier
        if architecture != 'unknown':
             # Try the architecture-specific key
             ctx_key = f'{architecture}.context_length'
             arch_ctx_val = get_scalar_metadata(raw_metadata, ctx_key) # Use the helper
             if arch_ctx_val is not None:
                 try:
                     max_ctx = int(arch_ctx_val)
                 except (ValueError, TypeError):
                     logging.warning(f"Could not convert architecture-specific context_length '{arch_ctx_val}' to integer for {model_path}. Using default 4096.")
                     max_ctx = 4096 # Ensure it's the default if conversion fails
             else:
                 logging.warning(f"Architecture-specific context_length key '{ctx_key}' not found for {model_path}. Using default 4096.")
                 max_ctx = 4096 # Ensure it's the default if key is not found
        else:
             logging.warning(f"Architecture is unknown for {model_path}. Cannot determine architecture-specific context_length. Using default 4096.")
             max_ctx = 4096 # Ensure it's the default if architecture is unknown


        # Ensure max_ctx is an integer (final check)
        if not isinstance(max_ctx, int):
             logging.warning(f"Final max_context_length '{max_ctx}' is not an integer for {model_path}. Defaulting to 4096.")
             max_ctx = 4096


        # Construct the LM Studio format dictionary
        lmstudio_format = {
            "id": model_id,
            "object": obj_type,
            "type": model_type,
            "publisher": publisher,
            "arch": architecture,
            "compatibility_type": compatibility_type,
            "quantization": quantization,
            # State will be added later based on runtime status
            "max_context_length": max_ctx
        }

        # Add LM Studio format fields to final_metadata
        final_metadata.update(lmstudio_format)

        # Add raw_metadata to final_metadata
        final_metadata["raw_metadata"] = raw_metadata

        logging.info(f"Successfully extracted metadata for {model_path}")
        # --- Add debug logging for the return value ---
        logging.debug(f"Successfully extracted metadata for {model_path}: {final_metadata}")
        # --- End debug logging ---
        return final_metadata

    except Exception as e:
        # Add traceback to the main extraction error logging
        logging.error(f"Error extracting GGUF metadata from {model_path}: {e}\n{traceback.format_exc()}")
        # --- Add debug logging for the return value on error ---
        logging.debug(f"Metadata extraction failed for {model_path}. Returning None.")
        # --- End debug logging ---
        return None

def get_model_lmstudio_format(model_name: str, model_path: str, model_config: Dict[str, Any], is_running: bool) -> Optional[Dict[str, Any]]:
    """
    Gets model metadata in LM Studio format, using cache if available.
    Includes the current running state.
    """
    if not GGUF_AVAILABLE:
        # Return a minimal structure if GGUF library is not available
        return {
            "id": model_name if "model_id" not in model_config else model_config["model_id"],
            "object": "model",
            "type": "llm", # Assume LLM if no metadata
            "publisher": "local",
            "arch": "unknown",
            "compatibility_type": "unknown",
            "quantization": "unknown",
            "state": "loaded" if is_running else "not-loaded",
            "max_context_length": 4096 # Default if no metadata
        }

    # Use file size for caching
    file_size = get_file_size(model_path)
    if file_size is None:
        logging.error(f"Could not get size for {model_path}. Cannot use cache.")
        # Fallback to extracting without caching if size retrieval fails
        metadata = extract_gguf_metadata(model_path, model_config)
        if metadata:
             metadata["state"] = "loaded" if is_running else "not-loaded"
             # Add file size to the metadata, formatted human-readable
             metadata["size"] = file_size
        return metadata


    cached_metadata = load_metadata_from_cache(model_name, file_size)

    if cached_metadata:
        # Update the state based on current runtime status
        cached_metadata["state"] = "loaded" if is_running else "not-loaded"

        # Ensure model_id from model_config takes precedence if provided
        if "model_id" in model_config and model_config["model_id"]:
            config_model_id = model_config["model_id"]
            # Check if cached_metadata has an 'id' and if it differs
            if cached_metadata.get("id") != config_model_id:
                logging.info(
                    f"Model '{model_name}': Config model_id '{config_model_id}' "
                    f"differs from cached id '{cached_metadata.get('id')}'. Overriding and updating cache."
                )
                cached_metadata["id"] = config_model_id
                # Re-save the updated metadata to cache.
                # file_size was determined near the start of this function (line 412).
                save_metadata_to_cache(model_name, file_size, cached_metadata)
        
        return cached_metadata
    else:
        logging.info(f"Cache miss or invalid for {model_name} (size: {file_size}). Extracting metadata...")
        extracted_metadata = extract_gguf_metadata(model_path, model_config)
        if extracted_metadata:
            # Add state and save to cache
            extracted_metadata["state"] = "loaded" if is_running else "not-loaded"
            # Add file size to the metadata, formatted human-readable
            extracted_metadata["size"] = file_size
            save_metadata_to_cache(model_name, file_size, extracted_metadata)
            return extracted_metadata
        else:
            logging.error(f"Failed to extract metadata for {model_name} at {model_path}")
            # Return a minimal structure if extraction fails
            return {
                "id": model_name if "model_id" not in model_config else model_config["model_id"],
                "object": "model",
                "type": "llm", # Assume LLM if no metadata
                "publisher": "local",
                "arch": "unknown",
                "compatibility_type": "unknown",
                "quantization": "unknown",
                "state": "loaded" if is_running else "not-loaded",
                "max_context_length": 4096 # Default if no metadata
            }

def get_all_models_lmstudio_format(models_config: Dict[str, Dict[str, Any]], is_model_running_callback) -> List[Dict[str, Any]]:
    """
    Gets metadata for all configured models in LM Studio format.
    Uses the running status callback to determine the state.
    """
    all_models_data = []
    for model_name, model_config in models_config.items():
        model_path = model_config.get("model_path")
        if not model_path:
            logging.warning(f"Model '{model_name}' has no 'model_path' in config. Skipping metadata.")
            continue

        is_running = is_model_running_callback(model_name)
        metadata = get_model_lmstudio_format(model_name, model_path, model_config, is_running)
        if metadata:
            all_models_data.append(metadata)

    return all_models_data

def get_single_model_lmstudio_format(model_name: str, models_config: Dict[str, Dict[str, Any]], is_model_running_callback) -> Optional[Dict[str, Any]]:
    """
    Gets metadata for a single configured model in LM Studio format.
    Uses the running status callback to determine the state.
    """
    model_config = models_config.get(model_name)
    if not model_config:
        return None # Model not found

    model_path = model_config.get("model_path")
    if not model_path:
        logging.warning(f"Model '{model_name}' has no 'model_path' in config. Cannot get metadata.")
        return None

    is_running = is_model_running_callback(model_name)
    metadata = get_model_lmstudio_format(model_name, model_path, model_config, is_running)

    return metadata

# Ensure cache directory exists on import
ensure_cache_dir_exists()

def get_model_name_to_id_mapping(models_config: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Gets a mapping of model names to their IDs from the metadata.
    """
    mapping = {}
    for model_name, model_config in models_config.items():
        model_path = model_config.get("model_path")
        if model_path:
            metadata = get_model_lmstudio_format(model_name, model_path, model_config, False)
            if metadata:
                mapping[model_name] = metadata["id"]
    return mapping
