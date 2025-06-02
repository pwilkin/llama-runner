import asyncio

import logging
import traceback
import json
from typing import Dict, Any, Callable, Optional

# Removed: from litellm.proxy.proxy_server import app
# Standard library imports
import httpx

# Third-party imports
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute # Import APIRoute for isinstance check
import uvicorn

from PySide6.QtCore import QThread, Slot # Import QTimer for potential use

from llama_runner import gguf_metadata # Import the new metadata module
from llama_runner.config_loader import calculate_system_fingerprint

# Configure logging (already done in main.py for configurable levels)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Create our own FastAPI app instance ---
app = FastAPI()
# --- End create app instance ---


# Define standalone handlers that access state via app.state
@app.get("/api/v0/models")
async def _get_lmstudio_models_handler(request: Request):
    """Handler for GET /api/v0/models"""
    # Access state from the request's app instance
    # Access state from the request's app instance
    all_models_config = request.app.state.all_models_config # Use all_models_config
    is_model_running_callback = request.app.state.is_model_running_callback

    if not gguf_metadata.GGUF_AVAILABLE:
         return JSONResponse(content={"error": "GGUF library not available for metadata extraction."}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        all_models_data = gguf_metadata.get_all_models_lmstudio_format(
            all_models_config, is_model_running_callback
        )
        # Add capabilities for models with has_tools
        id_mapping = gguf_metadata.get_model_name_to_id_mapping(all_models_config)
        id_to_internal = {v: k for k, v in id_mapping.items()}
        for model in all_models_data:
            internal_name = id_to_internal.get(model['id'])
            if internal_name:
                model_config = all_models_config.get(internal_name, {})
                if model_config.get('has_tools'):
                    model['capabilities'] = ["tool_use"]
        return JSONResponse(content={
            "object": "list",
            "data": all_models_data
        })
    except Exception as e:
        logging.error(f"Error handling /api/v0/models: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error retrieving models metadata")

@app.get("/api/v0/models/{model_id}")
async def _get_lmstudio_model_handler(model_id: str, request: Request):
    """Handler for GET /api/v0/models/{model_id}"""
    # Access state from the request's app instance
    # Access state from the request's app instance
    all_models_config = request.app.state.all_models_config # Use all_models_config
    is_model_running_callback = request.app.state.is_model_running_callback

    if not gguf_metadata.GGUF_AVAILABLE:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GGUF library not available for metadata extraction.")

    try:
        # Find the model in the config by its LM Studio ID (which is the model_name from config)
        # Find the model in the config by its LM Studio ID (which is the model_name from config)
        model_data = gguf_metadata.get_single_model_lmstudio_format(
            model_id, all_models_config, is_model_running_callback # Use all_models_config
        )

        if model_data:
            return JSONResponse(content=model_data)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with id '{model_id}' not found")

    except Exception as e:
        logging.error(f"Error handling /api/v0/models/{model_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error retrieving model metadata")


# --- Handler for dynamic routing of /v1/* requests ---
async def _dynamic_route_v1_request_generator(
    request: Request,
    target_path: Optional[str] = None,
    body: Optional[dict] = None,
    body_bytes: Optional[bytes] = None
):
    """
    Intercepts /v1/* requests, ensures the target runner is running,
    and forwards the request to the runner's port, yielding the response chunks.
    This is an async generator function for streaming responses.
    """
    # Access state and callbacks from the request's app instance
    all_models_config = request.app.state.all_models_config # Use all_models_config
    runtimes_config = request.app.state.runtimes_config # Use runtimes_config
    get_runner_port_callback = request.app.state.get_runner_port_callback
    request_runner_start_callback = request.app.state.request_runner_start_callback
    prompt_logging_enabled = getattr(request.app.state, 'prompt_logging_enabled', False) # Safer access
    prompts_logger = getattr(request.app.state, 'prompts_logger', logging.getLogger()) # Safer access
    # Access the proxy thread instance to get the futures dictionary
    proxy_thread_instance = getattr(request.app.state, 'proxy_thread_instance', None)
    if not isinstance(proxy_thread_instance, FastAPIProxyThread):
        logging.error("proxy_thread_instance not found or not of type FastAPIProxyThread in app.state")
        yield f'data: {{"error": "Internal server error: Proxy thread not configured."}}\n\n'.encode('utf-8')
        return
    proxy_thread: FastAPIProxyThread = proxy_thread_instance


    # Extract the model name from the request body
    try:
        # If not provided, read the body here (for backward compatibility, but should always be passed in)
        if body is None or body_bytes is None:
            body_bytes = await request.body()
            body = {}
            if body_bytes:
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    body = None # Ensure body is None if JSON decoding fails
                    logging.warning(f"Could not decode request body as JSON for {request.url.path}")
                    yield b'data: {"error": "Invalid JSON request body."}\n\n'
                    return # Stop the generator
        
        model_name_from_request = None
        if isinstance(body, dict): # Check if body is a dict before calling .get()
            model_name_from_request = body.get("model") # This is the ID used by LM Studio (e.g., "vendor/model-name-gguf")
        
        if not model_name_from_request:
            logging.warning(f"Model name not found in request body for {request.url.path}")
            yield b'data: {"error": "Model name not specified in request body."}\n\n'
            return # Stop the generator

        # Log the incoming request if prompt logging is enabled
        if prompt_logging_enabled:
            try:
                # Log the request body as a JSON string
                prompts_logger.info(f"Request to {request.url.path} for model '{model_name_from_request}': {json.dumps(body)}")
            except Exception as log_e:
                logging.error(f"Error logging request body for {request.url.path}: {log_e}")

    except Exception as e:
        logging.error(f"Error reading request body or extracting model name: {e}\n{traceback.format_exc()}")
        yield f'data: {{"error": "Invalid request: {e}"}}\n\n'.encode('utf-8')
        return # Stop the generator

    logging.debug(f"Intercepted request for model (ID from request): {model_name_from_request} at path: {request.url.path}")
    logging.debug(f"Available models in all_models_config: {list(all_models_config.keys())}")
    logging.debug(f"Available runtimes in runtimes_config: {list(runtimes_config.keys())}")

    logging.debug(f"Intercepted request for model (ID from request): {model_name_from_request} at path: {request.url.path}")
    logging.debug(f"Available models in all_models_config: {list(all_models_config.keys())}")
    logging.debug(f"Available runtimes in runtimes_config: {list(runtimes_config.keys())}")

    # LM Studio uses an ID format (e.g., "vendor/model-file.gguf") in requests.
    # We need to map this ID back to our internal model name (the key in all_models_config).
    # The gguf_metadata.get_model_name_to_id_mapping uses all_models_config.
    id_to_internal_name_mapping = {v: k for k, v in gguf_metadata.get_model_name_to_id_mapping(all_models_config).items()}
    internal_model_name = id_to_internal_name_mapping.get(model_name_from_request)

    if not internal_model_name:
        # Fallback: if the request model_name is already an internal name (should not happen for LM Studio proxy)
        if model_name_from_request in all_models_config:
            internal_model_name = model_name_from_request
            logging.warning(f"Request model ID '{model_name_from_request}' matched an internal model name directly. This might indicate a misconfiguration or unexpected request format.")
        else:
            logging.warning(f"Request for unknown model ID: {model_name_from_request}. Could not map to an internal model name.")
            yield f"data: {{\"error\": \"Model ID '{model_name_from_request}' not found in configuration mapping.\"}}\n\n".encode('utf-8')
            return # Stop the generator
    
    logging.debug(f"Mapped request model ID '{model_name_from_request}' to internal model name '{internal_model_name}'")

    # Get the specific model's configuration details from all_models_config using the internal_model_name
    model_config_details = all_models_config.get(internal_model_name)
    runtime_name_for_model = model_config_details.get("llama_cpp_runtime") if model_config_details else None

    if not runtime_name_for_model:
        logging.warning(f"Runtime not defined for model '{internal_model_name}' (from request ID '{model_name_from_request}') in main configuration.")
        yield f"data: {{\"error\": \"Runtime not configured for model '{internal_model_name}'.\"}}\n\n".encode('utf-8')
        return

    # Get the runtime's configuration from runtimes_config
    runtime_details_from_config = runtimes_config.get(runtime_name_for_model)

    if not runtime_details_from_config:
        logging.warning(f"Configuration for runtime '{runtime_name_for_model}' (for model '{internal_model_name}') not found in runtimes configuration.")
        yield f"data: {{\"error\": \"Configuration for runtime '{runtime_name_for_model}' not found.\"}}\n\n".encode('utf-8')
        return

    # Conditionally remove 'tools' and 'tool_choice' from the request body
    if body_bytes and body: # Ensure body was successfully parsed
        if runtime_details_from_config.get("supports_tools") is False:
            tools_present_in_request = "tools" in body
            tool_choice_present_in_request = "tool_choice" in body

            if tools_present_in_request:
                body.pop("tools")
            if tool_choice_present_in_request:
                body.pop("tool_choice")

            if tools_present_in_request or tool_choice_present_in_request:
                logging.info(
                    f"Model '{internal_model_name}' (request ID: '{model_name_from_request}', runtime: '{runtime_name_for_model}') "
                    f"has supports_tools=False. Removed 'tools' and/or 'tool_choice' from request to {request.url.path}."
                )
                # Re-encode the modified body to body_bytes as it's used later for forwarding
                body_bytes = json.dumps(body).encode('utf-8')

    # Check if the runner is already running using the internal_model_name
    # Note: The 'model_name' variable used from here onwards for runner management
    # should be the 'internal_model_name'.
    model_name = internal_model_name # Ensure 'model_name' refers to the internal name for subsequent logic
    port = get_runner_port_callback(model_name)

    if port is None:
        # Runner is not running, request startup and wait
        logging.info(f"Runner for {model_name} not running. Requesting startup.")
        startup_timeout = 240 # Define startup_timeout before the try block
        try:
            # Request startup via the callback, which returns an asyncio.Future
            # Store the future locally in the proxy thread instance
            if model_name not in proxy_thread._runner_ready_futures or proxy_thread._runner_ready_futures[model_name].done():
                 logging.debug(f"Creating new startup future for {model_name}")
                 proxy_thread._runner_ready_futures[model_name] = request_runner_start_callback(model_name)
            else:
                 logging.debug(f"Using existing startup future for {model_name}")

            # Wait for the runner to become ready (Future to resolve)
            # Use a timeout to prevent infinite waiting
            port = await asyncio.wait_for(
                proxy_thread._runner_ready_futures[model_name],
                timeout=startup_timeout
            )
            logging.info(f"Runner for {model_name} is ready on port {port} after startup.")

        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for runner {model_name} to start after {startup_timeout} seconds.")
            # Clean up the future if it timed out
            if model_name in proxy_thread._runner_ready_futures and not proxy_thread._runner_ready_futures[model_name].done():
                 proxy_thread._runner_ready_futures[model_name].cancel() # Cancel the future
                 del proxy_thread._runner_ready_futures[model_name]
            yield f"data: {{\"error\": \"Timeout starting runner for model '{model_name}'.\"}}\n\n".encode('utf-8')
            return # Stop the generator
        except Exception as e:
            logging.error(f"Error during runner startup for {model_name}: {e}\n{traceback.format_exc()}")
            # Clean up the future if it failed
            if model_name in proxy_thread._runner_ready_futures and not proxy_thread._runner_ready_futures[model_name].done():
                 proxy_thread._runner_ready_futures[model_name].set_exception(e) # Set exception on the future
                 del proxy_thread._runner_ready_futures[model_name]
            yield f"data: {{\"error\": \"Error starting runner for model '{model_name}': {e}\"}}\n\n".encode('utf-8')
            return # Stop the generator

    else:
        logging.debug(f"Runner for {model_name} is already running on port {port}.")
        # If it was running, ensure its future is marked as done with the port
        # This handles cases where the proxy restarts but the runner is still alive
        if model_name not in proxy_thread._runner_ready_futures or not proxy_thread._runner_ready_futures[model_name].done():
             logging.debug(f"Creating completed future for already running runner {model_name}")
             future = asyncio.Future()
             future.set_result(port)
             proxy_thread._runner_ready_futures[model_name] = future
             # No need for timer cleanup here, as it's already running.
             # The future will be removed if the runner stops later.


    # Runner is ready and port is known. Forward the request.
    # Construct the target URL using the known port and the provided target_path or original request path
    path_to_use = target_path if target_path is not None else request.url.path
    target_url = f"http://127.0.0.1:{port}{path_to_use}"
    logging.debug(f"Target URL: {target_url}")
    logging.debug(f"Forwarding request for {model_name} to {target_url}")

    # Use httpx to forward the request and yield chunks directly
    async with httpx.AsyncClient() as client:
        response_chunks = [] # Buffer for response chunks
        try:
            # Reconstruct headers, removing host and potentially others that shouldn't be forwarded
            headers = dict(request.headers)
            headers.pop('host', None) # Remove host header
            # Remove content-length header as httpx will set it correctly based on the forwarded content
            headers.pop('content-length', None)
            # Forward the request, including method, URL path, headers, and body
            # Use the body_bytes read earlier
            async with client.stream(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes, # Pass the raw body bytes
                timeout=600.0, # Use a generous timeout for model responses
            ) as proxy_response:

                # Check if the response is streaming (e.g., Server-Sent Events)
                # This requires inspecting headers like 'content-type'
                content_type = proxy_response.headers.get('content-type', '').lower()
                is_streaming = 'text/event-stream' in content_type

                if is_streaming:
                    logging.debug(f"Forwarding streaming response from {target_url}")
                    # Load config and calculate fingerprint once for the stream
                    config = request.app.state.all_models_config
                    model_config = config.get(model_name, {})
                    system_fingerprint = calculate_system_fingerprint(model_config)

                    # Process and yield chunks, adding fingerprint if needed
                    async for chunk in proxy_response.aiter_bytes():
                        if prompt_logging_enabled:
                             response_chunks.append(chunk) # Buffer chunk for logging
                        try:
                            chunk_str = chunk.decode('utf-8').strip()
                            if chunk_str.startswith('data: '):
                                json_payload_str = chunk_str[len('data: '):].strip()
                                if json_payload_str == '[DONE]':
                                    yield chunk # Pass through the DONE signal
                                    continue
                                try:
                                    data_json = json.loads(json_payload_str)
                                    if 'system_fingerprint' not in data_json:
                                        data_json['system_fingerprint'] = system_fingerprint
                                        modified_chunk_str = f'data: {json.dumps(data_json)}\n\n'
                                        yield modified_chunk_str.encode('utf-8')
                                    else:
                                        yield chunk # Yield original chunk if fingerprint exists
                                except json.JSONDecodeError:
                                    logging.warning(f"Could not decode JSON from streaming chunk: {json_payload_str}")
                                    yield chunk # Yield original chunk if not valid JSON
                            else:
                                # Yield non-data lines (e.g., comments, empty lines) as is
                                yield chunk
                        except Exception as e:
                            logging.error(f"Error processing streaming chunk: {e}\n{traceback.format_exc()}")
                            yield chunk # Yield original chunk in case of processing error
                else:
                    logging.debug(f"Forwarding non-streaming response from {target_url}")
                    # For non-streaming responses, read the body and yield it as a single chunk
                    response_body = await proxy_response.aread() # Read the entire body asynchronously
                    if prompt_logging_enabled:
                         response_chunks.append(response_body) # Buffer the full body
                    try:
                        response_json = json.loads(response_body.decode('utf-8'))
                        # Check if 'system_fingerprint' is missing
                        if 'system_fingerprint' not in response_json:
                            # Load config and calculate fingerprint
                            config = request.app.state.all_models_config
                            model_config = config.get(model_name, {})
                            system_fingerprint = calculate_system_fingerprint(model_config)
                            # Add 'system_fingerprint' to the response JSON
                            response_json['system_fingerprint'] = system_fingerprint
                            # Re-encode the modified JSON
                            modified_response_body = json.dumps(response_json).encode('utf-8')
                        else:
                            modified_response_body = response_body
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON from runner response: {response_body.decode('utf-8')}")
                        modified_response_body = response_body # Use original if JSON decode fails

                    if proxy_response.status_code != 200:
                        # Handle non-200 responses
                        logging.error(f"Error response from {target_url}: {proxy_response.status_code} - {modified_response_body.decode('utf-8')}")
                    yield modified_response_body # Yield the modified (or original) body as one chunk

        except httpx.RequestError as e:
            logging.error(f"Error forwarding request to runner {model_name} on port {port}: {e}\n{traceback.format_exc()}")
            # Log the error response if prompt logging is enabled
            if prompt_logging_enabled:
                 prompts_logger.error(f"Error response from {target_url} for model '{model_name}': {e}")
            # In a generator, we can't raise HTTPException directly.
            # We could yield an error message or re-raise after the generator is consumed.
            # Yielding an error message in SSE format is a common pattern.
            yield f'data: {{"error": "Error communicating with runner for model \'{model_name}\': {e}"}}\n\n'.encode('utf-8')
            # No return here, let the generator finish naturally


        except Exception as e:
            logging.error(f"Unexpected error during request forwarding for {model_name}: {e}\n{traceback.format_exc()}")
            # Log the unexpected error if prompt logging is enabled
            if prompt_logging_enabled:
                 prompts_logger.error(f"Unexpected error processing request for model '{model_name}': {e}")
            # Handle unexpected errors similarly
            yield f'data: {{"error": "Internal error processing request for model \'{model_name}\': {e}"}}\n\n'.encode('utf-8')
            # No return here, let the generator finish naturally

        finally:
            # Log the complete response if prompt logging is enabled and response_chunks were collected
            if prompt_logging_enabled and response_chunks:
                 try:
                     full_response_bytes = b''.join(response_chunks)
                     # Attempt to decode and log as JSON if possible, otherwise log raw bytes
                     try:
                         full_response_json = json.loads(full_response_bytes.decode('utf-8'))
                         prompts_logger.info(f"Response from {target_url} for model '{model_name}': {json.dumps(full_response_json)}")
                     except json.JSONDecodeError:
                         # If not JSON, log the raw string or a truncated version
                         response_str = full_response_bytes.decode('utf-8', errors='replace')
                         prompts_logger.info(f"Raw response from {target_url} for model '{model_name}': {response_str[:500]}...") # Log first 500 chars
                 except Exception as log_e:
                     logging.error(f"Error logging response body for {request.url.path}: {log_e}")


# --- End handler for /v1/* requests ---


# --- Handlers for /api/v0/* proxying ---
# These handlers will call the _dynamic_route_v1_request_generator handler internally
@app.post("/api/v0/chat/completions")
async def _proxy_v0_chat_completions(request: Request):
    """Proxies /api/v0/chat/completions to /v1/chat/completions."""
    logging.debug("Proxying /api/v0/chat/completions to /v1/chat/completions")
    try:
        body_bytes = await request.body()
        body = {}
        if body_bytes:
            body = json.loads(body_bytes)
    except Exception as e:
        logging.error(f"Error reading request body in /api/v0/chat/completions: {e}\n{traceback.format_exc()}")
        body = {}
        body_bytes = b""
    return StreamingResponse(content=_dynamic_route_v1_request_generator(request, target_path="/v1/chat/completions", body=body, body_bytes=body_bytes))

@app.post("/api/v0/embeddings")
async def _proxy_v0_embeddings(request: Request):
    """Proxies /api/v0/embeddings to /v1/embeddings."""
    logging.debug("Proxying /api/v0/embeddings to /v1/embeddings")
    try:
        body_bytes = await request.body()
        body = {}
        if body_bytes:
            body = json.loads(body_bytes)
    except Exception as e:
        logging.error(f"Error reading request body in /api/v0/embeddings: {e}\n{traceback.format_exc()}")
        body = {}
        body_bytes = b""
    return StreamingResponse(content=_dynamic_route_v1_request_generator(request, target_path="/v1/embeddings", body=body, body_bytes=body_bytes))

@app.post("/api/v0/completions")
async def _proxy_v0_completions(request: Request):
    """Proxies /api/v0/completions to /v1/completions."""
    logging.debug("Proxying /api/v0/completions to /v1/completions")
    try:
        body_bytes = await request.body()
        body = {}
        if body_bytes:
            body = json.loads(body_bytes)
    except Exception as e:
        logging.error(f"Error reading request body in /api/v0/completions: {e}\n{traceback.format_exc()}")
        body = {}
        body_bytes = b""
    return StreamingResponse(content=_dynamic_route_v1_request_generator(request, target_path="/v1/completions", body=body, body_bytes=body_bytes))

# --- End handlers for /api/v0/* proxying ---


# --- Handler for /v1/models endpoint (OpenAI compatible) ---
@app.get("/v1/models")
async def _list_openai_models_handler(request: Request):
    """Handler for GET /v1/models, returns a simplified OpenAI-compatible list."""
    try:
        all_models_config = request.app.state.all_models_config
        # Get the mapping from internal name to LM Studio ID
        id_mapping = gguf_metadata.get_model_name_to_id_mapping(all_models_config)

        # Create the list of models in OpenAI format
        models_list = []
        for lmstudio_id in id_mapping.values():
            models_list.append({
                "id": lmstudio_id,
                "object": "model",
                "owned_by": "organization_owner" # Standard value for local models
            })

        return JSONResponse(content={
            "object": "list",
            "data": models_list
        })
    except Exception as e:
        logging.error(f"Error handling /v1/models: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error retrieving models list")

# --- End handler for /v1/models ---


# --- Add routes for /v1/* endpoints to be intercepted ---
# Use a path parameter to capture the rest of the path after /v1
# This allows intercepting /v1/chat/completions, /v1/completions, etc.
# The handler will then forward to the correct path on the runner.
# Note: This overrides LiteLLM's default handling for /v1 endpoints.
# If LiteLLM's internal routing is needed *after* runner startup, a different approach is required.
# This approach assumes we are completely bypassing LiteLLM's model routing for /v1.
# We need to add these routes *before* LiteLLM's default /v1 routes are potentially added
# if LiteLLM's app instance includes them by default.
# Inspecting `app.routes` might be necessary to ensure our routes take precedence.
# For simplicity, let's assume adding them here works.

# Check if specific /v1 routes exist before adding our dynamic handler
# This check is complex because LiteLLM might add routes like /v1/chat/completions directly.
# A simpler approach is to add our catch-all /v1/{path:path} route and ensure it's processed first.
# FastAPI processes routes in the order they are added.

# Let's add specific routes for the common endpoints first, then a catch-all if needed.
# This is safer than a broad catch-all if LiteLLM adds other /v1 routes we don't want to intercept.

# Add specific routes for common /v1 endpoints
# The handler will extract the model name from the request body
# Check if our dynamic handlers are already added to avoid duplicates on reload/restart
# This check is fragile, a better approach might be needed if routes are added dynamically elsewhere
# Note: /v1/models is handled separately above.
dynamic_v1_paths = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
# Check if the *handler function itself* is already associated with the path
# This is a more robust check than just checking the path string
# Use isinstance and APIRoute for Pylance
current_v1_handlers = {}
for route in app.routes:
    if isinstance(route, APIRoute) and route.path in dynamic_v1_paths:
        current_v1_handlers[route.path] = route.endpoint

# Add routes using the @app.post decorator
@app.post("/v1/chat/completions")
async def _v1_chat_completions_handler(request: Request):
    try:
        body_bytes = await request.body()
        body = {}
        if body_bytes:
            body = json.loads(body_bytes)
    except Exception as e:
        logging.error(f"Error reading request body in /v1/chat/completions: {e}\n{traceback.format_exc()}")
        body = {}
        body_bytes = b""
    return StreamingResponse(content=_dynamic_route_v1_request_generator(request, body=body, body_bytes=body_bytes))

@app.post("/v1/completions")
async def _v1_completions_handler(request: Request):
    try:
        body_bytes = await request.body()
        body = {}
        if body_bytes:
            body = json.loads(body_bytes)
    except Exception as e:
        logging.error(f"Error reading request body in /v1/completions: {e}\n{traceback.format_exc()}")
        body = {}
        body_bytes = b""
    return StreamingResponse(content=_dynamic_route_v1_request_generator(request, body=body, body_bytes=body_bytes))

@app.post("/v1/embeddings")
async def _v1_embeddings_handler(request: Request):
    try:
        body_bytes = await request.body()
        body = {}
        if body_bytes:
            body = json.loads(body_bytes)
    except Exception as e:
        logging.error(f"Error reading request body in /v1/embeddings: {e}\n{traceback.format_exc()}")
        body = {}
        body_bytes = b""
    return StreamingResponse(content=_dynamic_route_v1_request_generator(request, body=body, body_bytes=body_bytes))

logging.info("Added dynamic routing handlers for /v1/chat/completions, /v1/completions, /v1/embeddings.")


# If needed, add a catch-all for other /v1 paths, but be cautious
# @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
# async def _v1_catch_all_handler(request: Request):
#     return await _dynamic_route_v1_request_generator(request)
# logging.info("Added catch-all dynamic routing handler for /v1/*.")


# --- End add routes ---


class FastAPIProxyThread(QThread): # Renamed class
    """
    QThread to run the FastAPI proxy in a separate thread. # Updated description
    Handles LM Studio API emulation and runner management requests.
    """
    # Define signals if needed (e.g., started, stopped, error)
    # started = Signal()
    # stopped = Signal()
    # error = Signal(str)

    def __init__(self,
                 all_models_config: Dict[str, Dict[str, Any]], # Renamed models_config
                 runtimes_config: Dict[str, Dict[str, Any]], # Added runtimes_config
                 is_model_running_callback: Callable[[str], bool],
                 get_runner_port_callback: Callable[[str], Optional[int]],
                 request_runner_start_callback: Callable[[str], asyncio.Future], # Callback now returns Future
                 prompt_logging_enabled: bool, # Add prompt logging flag
                 prompts_logger: logging.Logger, # Add prompts logger instance
                 api_key: Optional[str] = None): # Changed to Optional[str]
        super().__init__()
        self.all_models_config = all_models_config # Store all_models_config
        self.runtimes_config = runtimes_config # Store runtimes_config
        self.is_model_running_callback = is_model_running_callback
        self.get_runner_port_callback = get_runner_port_callback
        self.request_runner_start_callback = request_runner_start_callback # Store the callback
        self.prompt_logging_enabled = prompt_logging_enabled # Store the flag
        self.prompts_logger = prompts_logger # Store the logger instance
        self.api_key = api_key
        self.is_running = False
        self._uvicorn_server = None
        # self._temp_config_path = None # No longer needed

        # Dictionary to hold asyncio Futures for runners that are starting
        # This dictionary is managed by MainWindow, but the proxy thread needs
        # a reference to it or a way to access/manage the Futures.
        # Let's pass the reference to the dictionary from MainWindow.
        # Or, the callbacks from MainWindow can manage the Futures directly.
        # The current design has MainWindow manage the Futures and return them
        # from request_runner_start_callback. The proxy thread will store
        # a *local* copy of the Futures it's currently waiting on.
        self._runner_ready_futures: Dict[str, asyncio.Future] = {}

        # Connect to signals from MainWindow (MainWindow connects these signals to our slots)
        # Signals: runner_port_ready_for_proxy, runner_stopped_for_proxy
        # Slots: on_runner_port_ready, on_runner_stopped


    @Slot(str, int)
    def on_runner_port_ready(self, model_name: str, port: int):
        """Slot to handle runner_port_ready_for_proxy signal from MainWindow."""
        logging.debug(f"Proxy thread received runner_port_ready for {model_name} on port {port}")
        # When a runner is ready, resolve its corresponding Future in our local dictionary
        if model_name in self._runner_ready_futures and not self._runner_ready_futures[model_name].done():
            logging.debug(f"Resolving local runner_ready_future for {model_name} with port {port}")
            self._runner_ready_futures[model_name].set_result(port)
        elif model_name in self._runner_ready_futures and self._runner_ready_futures[model_name].done():
             logging.warning(f"Received runner_port_ready for {model_name}, but local Future was already done.")
        else:
             logging.info(f"Received runner_port_ready for {model_name}, but no pending local Future found. This can occur if the runner was started outside the proxy's request flow or the Future was already resolved. No action needed.")


    @Slot(str)
    def on_runner_stopped(self, model_name: str):
        """Slot to handle runner_stopped_for_proxy signal from MainWindow."""
        logging.debug(f"Proxy thread received runner_stopped for {model_name}")
        # If a runner stops, cancel or set exception on its Future in our local dictionary
        if model_name in self._runner_ready_futures:
             if not self._runner_ready_futures[model_name].done():
                 logging.debug(f"Setting exception on local runner_ready_future for {model_name} due to stop.")
                 self._runner_ready_futures[model_name].set_exception(RuntimeError(f"Runner for {model_name} stopped unexpectedly."))
             # Remove the future from the local dictionary
             del self._runner_ready_futures[model_name]
             logging.debug(f"Cleaned up local runner_ready_future for {model_name}")
        else:
             logging.debug(f"Received runner_stopped for {model_name}, but no local Future found.")


    def run(self):
        """
        Runs the FastAPI proxy in the thread using Uvicorn.
        """
        self.is_running = True
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # --- Set the proxy thread instance on the app state ---
            # This allows the standalone handlers to access the thread's local state (like _runner_ready_futures)
            app.state.proxy_thread_instance = self
            # --- End set instance ---

            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            logging.error(f"Unexpected error in FastAPIProxyThread run: {e}\n{traceback.format_exc()}")
        finally:
            self.is_running = False
            # Clean up the temporary config file (no longer generated)
            # if self._temp_config_path and os.path.exists(self._temp_config_path):
            #      try:
            #          os.unlink(self._temp_config_path)
            #          logging.info(f"Cleaned up temporary LiteLLM config file: {self._temp_config_path}")
            #      except OSError as e:
            #          logging.error(f"Error cleaning up temporary LiteLLM config file {self._temp_config_path}: {e}")
            # self._temp_config_path = None # Clear the path

            # Log proxy stop using the prompts logger if enabled
            if self.prompt_logging_enabled:
                 self.prompts_logger.info("FastAPI Proxy thread stopping.")

            # --- Clean up the proxy thread instance from app state ---
            if hasattr(app.state, 'proxy_thread_instance'):
                 del app.state.proxy_thread_instance
            # --- End cleanup ---

            if hasattr(self, 'loop') and self.loop.is_running():
                 self.loop.stop()
            if hasattr(self, 'loop') and not self.loop.is_closed():
                 self.loop.close()


    async def run_async(self):
        """
        Asynchronous part of the proxy runner.
        Starts the Uvicorn server for the FastAPI app.
        """
        print("Starting FastAPI Proxy...")
        try:
            # LiteLLM config generation removed.
            # The FastAPI app 'app' is already created at the module level.
            # Routes are added at the module level.

            # 4. Start the proxy embedded via Uvicorn
            # Set state on the global app instance BEFORE creating the server
            # This state will be accessible by the standalone handler functions
            app.state.all_models_config = self.all_models_config # Pass all_models_config
            app.state.runtimes_config = self.runtimes_config # Pass runtimes_config
            app.state.is_model_running_callback = self.is_model_running_callback
            app.state.get_runner_port_callback = self.get_runner_port_callback # Pass the new callback
            app.state.request_runner_start_callback = self.request_runner_start_callback # Pass the new callback
            app.state.prompt_logging_enabled = self.prompt_logging_enabled # Set prompt logging flag on state
            app.state.prompts_logger = self.prompts_logger # Set prompts logger on state
            # Extract metadata for all models and store it in app.state.models_metadata
            # Note: get_all_models_lmstudio_format expects the main models config (all_models_config)
            app.state.models_metadata = gguf_metadata.get_all_models_lmstudio_format(
                self.all_models_config, self.is_model_running_callback
            )

            # Use port 1234 as required
            uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=1234, reload=False)
            self._uvicorn_server = uvicorn.Server(uvicorn_config)

            print("FastAPI Proxy listening on http://127.0.0.1:1234")
            logging.info("FastAPI Proxy listening on http://127.0.0.1:1234")

            # Log proxy start using the prompts logger if enabled
            if self.prompt_logging_enabled:
                 self.prompts_logger.info("FastAPI Proxy thread started and listening on http://127.0.0.1:1234")

            # This call is blocking until the server stops
            await self._uvicorn_server.serve()

        except Exception as e:
            print(f"Error starting FastAPI Proxy: {e}")
            logging.error(f"Error starting FastAPI Proxy: {e}\n{traceback.format_exc()}")
            # Emit error signal if needed
            # self.error.emit(str(e))
        finally:
            # Cleanup happens in run()'s finally block now
            print("FastAPI Proxy stopped.")
            logging.info("FastAPI Proxy stopped.")


    def stop(self):
        """
        Signals the FastAPI proxy thread to stop.
        """
        self.is_running = False
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            # Note: Stopping uvicorn gracefully can be tricky in an embedded context.
            # This might not immediately stop the serve() call.
            # A more robust stop might involve sending a signal or using a shutdown event.
            # For now, setting should_exit is the standard uvicorn way.
