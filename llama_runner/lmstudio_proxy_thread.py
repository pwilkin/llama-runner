import asyncio

import logging
import traceback
import json
from typing import Dict, Any, Callable, Optional, AsyncGenerator

# Removed: from litellm.proxy.proxy_server import app
# Standard library imports
import httpx

# Third-party imports
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute # Import APIRoute for isinstance check
import uvicorn


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

# --- New function for non-streaming responses ---
async def _fetch_non_streaming_v1_response(
    request: Request,
    target_path: Optional[str] = None,
    body: Optional[dict] = None,
    body_bytes: Optional[bytes] = None
) -> Dict[str, Any] | list[Any]:
    """
    Handles non-streaming /v1/* requests. It ensures the target runner is running,
    forwards the request, and returns the complete response as a dictionary.
    """
    # Access state and callbacks from the request's app instance
    all_models_config = request.app.state.all_models_config
    runtimes_config = request.app.state.runtimes_config
    get_runner_port_callback = request.app.state.get_runner_port_callback
    request_runner_start_callback = request.app.state.request_runner_start_callback
    prompt_logging_enabled = getattr(request.app.state, 'prompt_logging_enabled', False)
    prompts_logger = getattr(request.app.state, 'prompts_logger', logging.getLogger())
    proxy_thread_instance = getattr(request.app.state, 'proxy_thread_instance', None)

    if not proxy_thread_instance:
        logging.error("proxy_thread_instance not found in app.state")
        return {"error": {"message": "Internal server error: Proxy not configured.", "type": "internal_error"}}
    # The instance is the LMStudioProxyServer itself.
    proxy_server = proxy_thread_instance


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
                    body = None
                    logging.warning(f"Could not decode request body as JSON for {request.url.path}")
                    return {"error": {"message": "Invalid JSON request body.", "type": "invalid_request_error"}}
        
        model_name_from_request = None
        if isinstance(body, dict):
            model_name_from_request = body.get("model")
        
        if not model_name_from_request:
            logging.warning(f"Model name not found in request body for {request.url.path}")
            return {"error": {"message": "Model name not specified in request body.", "type": "invalid_request_error"}}

        # Log the incoming request if prompt logging is enabled
        if prompt_logging_enabled:
            try:
                # Log the request body as a JSON string
                prompts_logger.info(f"Request to {request.url.path} for model '{model_name_from_request}': {json.dumps(body)}")
            except Exception as log_e:
                logging.error(f"Error logging request body for {request.url.path}: {log_e}")

    except Exception as e:
        logging.error(f"Error reading request body or extracting model name: {e}\n{traceback.format_exc()}")
        return {"error": {"message": f"Invalid request: {e}", "type": "invalid_request_error"}}

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
            return {"error": {"message": f"Model ID '{model_name_from_request}' not found in configuration mapping.", "type": "invalid_request_error"}}
    
    logging.debug(f"Mapped request model ID '{model_name_from_request}' to internal model name '{internal_model_name}'")

    # Get the specific model's configuration details from all_models_config using the internal_model_name
    model_config_details = all_models_config.get(internal_model_name)
    runtime_name_for_model = model_config_details.get("llama_cpp_runtime") if model_config_details else None

    if not runtime_name_for_model:
        logging.warning(f"Runtime not defined for model '{internal_model_name}' (from request ID '{model_name_from_request}') in main configuration.")
        return {"error": {"message": f"Runtime not configured for model '{internal_model_name}'.", "type": "configuration_error"}}

    # Get the runtime's configuration from runtimes_config
    runtime_details_from_config = runtimes_config.get(runtime_name_for_model)

    if not runtime_details_from_config:
        logging.warning(f"Configuration for runtime '{runtime_name_for_model}' (for model '{internal_model_name}') not found in runtimes configuration.")
        return {"error": {"message": f"Configuration for runtime '{runtime_name_for_model}' not found.", "type": "configuration_error"}}

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
            if model_name not in proxy_server._runner_ready_futures or proxy_server._runner_ready_futures[model_name].done():
                 logging.debug(f"Creating new startup future for {model_name}")
                 proxy_server._runner_ready_futures[model_name] = request_runner_start_callback(model_name)
            else:
                 logging.debug(f"Using existing startup future for {model_name}")

            # Wait for the runner to become ready (Future to resolve)
            # Use a timeout to prevent infinite waiting
            port = await asyncio.wait_for(
                proxy_server._runner_ready_futures[model_name],
                timeout=startup_timeout
            )
            logging.info(f"Runner for {model_name} is ready on port {port} after startup.")

        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for runner {model_name} to start after {startup_timeout} seconds.")
            # Clean up the future if it timed out
            if model_name in proxy_server._runner_ready_futures and not proxy_server._runner_ready_futures[model_name].done():
                 proxy_server._runner_ready_futures[model_name].cancel()
                 del proxy_server._runner_ready_futures[model_name]
            return {"error": {"message": f"Timeout starting runner for model '{model_name}'.", "type": "runner_startup_error"}}
        except Exception as e:
            logging.error(f"Error during runner startup for {model_name}: {e}\n{traceback.format_exc()}")
            if model_name in proxy_server._runner_ready_futures and not proxy_server._runner_ready_futures[model_name].done():
                 proxy_server._runner_ready_futures[model_name].set_exception(e)
                 del proxy_server._runner_ready_futures[model_name]
            return {"error": {"message": f"Error starting runner for model '{model_name}': {e}", "type": "runner_startup_error"}}

    else:
        logging.debug(f"Runner for {model_name} is already running on port {port}.")
        # If it was running, ensure its future is marked as done with the port
        # This handles cases where the proxy restarts but the runner is still alive
        if model_name not in proxy_server._runner_ready_futures or not proxy_server._runner_ready_futures[model_name].done():
             logging.debug(f"Creating completed future for already running runner {model_name}")
             future = asyncio.Future()
             future.set_result(port)
             proxy_server._runner_ready_futures[model_name] = future
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
                is_backend_streaming = 'text/event-stream' in content_type

                config = request.app.state.all_models_config
                model_config_for_fingerprint = config.get(model_name, {})
                system_fingerprint = calculate_system_fingerprint(model_config_for_fingerprint)

                if is_backend_streaming:
                    logging.debug(f"Collecting streaming response from {target_url} for non-streaming client (SSE -> Full)")
                    # collected_data_chunks = [] # F841: local variable 'collected_data_chunks' is assigned to but never used
                    final_response_obj = {"id": "chatcmpl-default", "object": "chat.completion", "created": int(asyncio.get_event_loop().time()), "model": model_name_from_request, "choices": []}
                    current_choice = {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}
                    has_content = False

                    async for chunk in proxy_response.aiter_bytes():
                        if prompt_logging_enabled:
                            response_chunks.append(chunk)
                        chunk_str = chunk.decode('utf-8').strip()
                        if chunk_str.startswith('data: '):
                            json_payload_str = chunk_str[len('data: '):].strip()
                            if json_payload_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(json_payload_str)
                                if data_json.get("choices"):
                                    delta = data_json["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"] is not None:
                                        current_choice["message"]["content"] += delta["content"]
                                        has_content = True
                                    if data_json["choices"][0].get("finish_reason"):
                                        current_choice["finish_reason"] = data_json["choices"][0]["finish_reason"]
                            except json.JSONDecodeError:
                                logging.warning(f"Could not decode JSON from streaming chunk for non-streaming client: {json_payload_str}")

                    if has_content or current_choice["finish_reason"]:
                        final_response_obj["choices"].append(current_choice)

                    if 'system_fingerprint' not in final_response_obj:
                        final_response_obj['system_fingerprint'] = system_fingerprint

                    return final_response_obj
                else: # Backend is not streaming, client wants full (ideal case for non-streaming)
                    logging.debug(f"Forwarding non-streaming response from {target_url} to non-streaming client (Full -> Full)")
                    response_body = await proxy_response.aread()
                    if prompt_logging_enabled:
                        response_chunks.append(response_body)
                    try:
                        response_json = json.loads(response_body.decode('utf-8'))
                        if (not isinstance(response_json, list)) and 'system_fingerprint' not in response_json:
                            response_json['system_fingerprint'] = system_fingerprint

                        if proxy_response.status_code != 200:
                            logging.error(f"Error response from {target_url}: {proxy_response.status_code} - {response_json}")
                            return response_json
                        return response_json
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON from runner response for non-streaming client: {response_body.decode('utf-8')}")
                        return {"error": {"message": "Runner returned non-JSON response.", "type": "runner_error", "details": response_body.decode('utf-8', errors='replace')[:500]}}

        except httpx.RequestError as e:
            logging.error(f"Error forwarding request to runner {model_name} on port {port}: {e}\n{traceback.format_exc()}")
            error_payload = {"error": {"message": f"Error communicating with runner for model '{model_name}': {e}", "type": "runner_communication_error"}}
            if prompt_logging_enabled:
                 prompts_logger.error(f"Error response from {target_url} for model '{model_name}': {e}")
            return error_payload

        except asyncio.TimeoutError as e:
            logging.error(f"Timeout during request forwarding for {model_name} to {target_url}: {e}\n{traceback.format_exc()}")
            error_payload = {"error": {"message": f"Timeout processing request for model '{model_name}'.", "type": "request_timeout_error"}}
            if prompt_logging_enabled:
                 prompts_logger.error(f"Timeout processing request for model '{model_name}': {e}")
            return error_payload

        except Exception as e:
            logging.error(f"Unexpected error during request forwarding for {model_name}: {e}\n{traceback.format_exc()}")
            error_payload = {"error": {"message": f"Internal error processing request for model '{model_name}': {e}", "type": "internal_error"}}
            if prompt_logging_enabled:
                 prompts_logger.error(f"Unexpected error processing request for model '{model_name}': {e}")
            return error_payload
        finally:
            if prompt_logging_enabled and response_chunks:
                 try:
                     full_response_bytes = b''.join(response_chunks)
                     try:
                         full_response_json = json.loads(full_response_bytes.decode('utf-8'))
                         prompts_logger.info(f"Response from {target_url} for model '{model_name}': {json.dumps(full_response_json)}")
                     except json.JSONDecodeError:
                         response_str = full_response_bytes.decode('utf-8', errors='replace')
                         prompts_logger.info(f"Raw response from {target_url} for model '{model_name}': {response_str[:500]}...")
                 except Exception as log_e:
                     logging.error(f"Error logging response body for {request.url.path}: {log_e}")

# --- End of _fetch_non_streaming_v1_response ---


# --- Modified generator for streaming responses ONLY ---
async def _dynamic_route_v1_request_generator(
    request: Request,
    target_path: Optional[str] = None,
    body: Optional[dict] = None,
    body_bytes: Optional[bytes] = None
) -> AsyncGenerator[bytes, None]: # Explicitly an AsyncGenerator
    """
    Intercepts /v1/* requests, ensures the target runner is running,
    and forwards the request to the runner's port, yielding the response chunks.
    This function ONLY handles streaming responses.
    """
    all_models_config = request.app.state.all_models_config
    runtimes_config = request.app.state.runtimes_config
    get_runner_port_callback = request.app.state.get_runner_port_callback
    request_runner_start_callback = request.app.state.request_runner_start_callback
    prompt_logging_enabled = getattr(request.app.state, 'prompt_logging_enabled', False)
    prompts_logger = getattr(request.app.state, 'prompts_logger', logging.getLogger())
    proxy_thread_instance = getattr(request.app.state, 'proxy_thread_instance', None)

    if not proxy_thread_instance:
        logging.error("proxy_thread_instance not found in app.state")
        error_payload = {"error": {"message": "Internal server error: Proxy not configured.", "type": "internal_error"}}
        yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
        return
    # The instance is the LMStudioProxyServer itself.
    proxy_server = proxy_thread_instance

    try:
        if body is None or body_bytes is None:
            body_bytes = await request.body()
            body = {}
            if body_bytes:
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    body = None
                    logging.warning(f"Could not decode request body as JSON for {request.url.path}")
                    error_payload = {"error": {"message": "Invalid JSON request body.", "type": "invalid_request_error"}}
                    yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
                    return

        model_name_from_request = None
        if isinstance(body, dict):
            model_name_from_request = body.get("model")

        if not model_name_from_request:
            logging.warning(f"Model name not found in request body for {request.url.path}")
            error_payload = {"error": {"message": "Model name not specified in request body.", "type": "invalid_request_error"}}
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return

        if prompt_logging_enabled:
            try:
                prompts_logger.info(f"Request to {request.url.path} for model '{model_name_from_request}': {json.dumps(body)}")
            except Exception as log_e:
                logging.error(f"Error logging request body for {request.url.path}: {log_e}")

    except Exception as e:
        logging.error(f"Error reading request body or extracting model name: {e}\n{traceback.format_exc()}")
        error_payload = {"error": {"message": f"Invalid request: {e}", "type": "invalid_request_error"}}
        yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
        return

    id_to_internal_name_mapping = {v: k for k, v in gguf_metadata.get_model_name_to_id_mapping(all_models_config).items()}
    internal_model_name = id_to_internal_name_mapping.get(model_name_from_request)

    if not internal_model_name:
        if model_name_from_request in all_models_config:
            internal_model_name = model_name_from_request
            logging.warning(f"Request model ID '{model_name_from_request}' matched an internal model name directly.")
        else:
            logging.warning(f"Request for unknown model ID: {model_name_from_request}.")
            error_payload = {"error": {"message": f"Model ID '{model_name_from_request}' not found in configuration mapping.", "type": "invalid_request_error"}}
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return

    model_config_details = all_models_config.get(internal_model_name)
    runtime_name_for_model = model_config_details.get("llama_cpp_runtime") if model_config_details else None

    if not runtime_name_for_model:
        logging.warning(f"Runtime not defined for model '{internal_model_name}'.")
        error_payload = {"error": {"message": f"Runtime not configured for model '{internal_model_name}'.", "type": "configuration_error"}}
        yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
        return

    runtime_details_from_config = runtimes_config.get(runtime_name_for_model)
    if not runtime_details_from_config:
        logging.warning(f"Configuration for runtime '{runtime_name_for_model}' not found.")
        error_payload = {"error": {"message": f"Configuration for runtime '{runtime_name_for_model}' not found.", "type": "configuration_error"}}
        yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
        return

    if body_bytes and body:
        if runtime_details_from_config.get("supports_tools") is False:
            if "tools" in body or "tool_choice" in body:
                body.pop("tools", None)
                body.pop("tool_choice", None)
                logging.info(f"Model '{internal_model_name}' has supports_tools=False. Removed 'tools'/'tool_choice'.")
                body_bytes = json.dumps(body).encode('utf-8')

    model_name = internal_model_name
    port = get_runner_port_callback(model_name)

    if port is None:
        logging.info(f"Runner for {model_name} not running. Requesting startup.")
        startup_timeout = 240
        try:
            if model_name not in proxy_server._runner_ready_futures or proxy_server._runner_ready_futures[model_name].done():
                 proxy_server._runner_ready_futures[model_name] = request_runner_start_callback(model_name)

            port = await asyncio.wait_for(proxy_server._runner_ready_futures[model_name], timeout=startup_timeout)
            logging.info(f"Runner for {model_name} is ready on port {port} after startup.")
        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for runner {model_name} to start.")
            if model_name in proxy_server._runner_ready_futures and not proxy_server._runner_ready_futures[model_name].done():
                 proxy_server._runner_ready_futures[model_name].cancel()
                 del proxy_server._runner_ready_futures[model_name]
            error_payload = {"error": {"message": f"Timeout starting runner for model '{model_name}'.", "type": "runner_startup_error"}}
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return
        except Exception as e:
            logging.error(f"Error during runner startup for {model_name}: {e}\n{traceback.format_exc()}")
            if model_name in proxy_server._runner_ready_futures and not proxy_server._runner_ready_futures[model_name].done():
                 proxy_server._runner_ready_futures[model_name].set_exception(e)
                 del proxy_server._runner_ready_futures[model_name]
            error_payload = {"error": {"message": f"Error starting runner for model '{model_name}': {e}", "type": "runner_startup_error"}}
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return
    else:
        logging.debug(f"Runner for {model_name} is already running on port {port}.")
        if model_name not in proxy_server._runner_ready_futures or not proxy_server._runner_ready_futures[model_name].done():
             future = asyncio.Future()
             future.set_result(port)
             proxy_server._runner_ready_futures[model_name] = future

    path_to_use = target_path if target_path is not None else request.url.path
    target_url = f"http://127.0.0.1:{port}{path_to_use}"
    logging.debug(f"Streaming: Target URL: {target_url}")

    async with httpx.AsyncClient() as client:
        response_chunks = []
        try:
            headers = dict(request.headers)
            headers.pop('host', None)
            headers.pop('content-length', None)

            async with client.stream(
                method=request.method, url=target_url, headers=headers, content=body_bytes, timeout=600.0
            ) as proxy_response:
                content_type = proxy_response.headers.get('content-type', '').lower()
                is_backend_streaming = 'text/event-stream' in content_type
                config = request.app.state.all_models_config
                model_config_for_fingerprint = config.get(model_name, {})
                system_fingerprint = calculate_system_fingerprint(model_config_for_fingerprint)

                if is_backend_streaming:
                    logging.debug(f"Streaming response from {target_url} to client (SSE -> SSE)")
                    async for chunk in proxy_response.aiter_bytes():
                        if prompt_logging_enabled:
                            response_chunks.append(chunk)
                        try:
                            chunk_str = chunk.decode('utf-8').strip()
                            if chunk_str.startswith('data: '):
                                json_payload_str = chunk_str[len('data: '):].strip()
                                if json_payload_str == '[DONE]':
                                    yield chunk
                                    continue
                                try:
                                    data_json = json.loads(json_payload_str)
                                    if 'system_fingerprint' not in data_json:
                                        data_json['system_fingerprint'] = system_fingerprint
                                        modified_chunk_str = f'data: {json.dumps(data_json)}\n\n'
                                        yield modified_chunk_str.encode('utf-8')
                                    else:
                                        yield chunk
                                except json.JSONDecodeError:
                                    logging.warning(f"Could not decode JSON from streaming chunk: {json_payload_str}")
                                    yield chunk
                            else:
                                yield chunk
                        except Exception as e: # pylint: disable=broad-except
                            logging.error(f"Error processing streaming chunk: {e}\n{traceback.format_exc()}")
                            yield chunk
                else: # Backend is not streaming, but client wants stream
                    logging.debug(f"Streaming response from {target_url} to client (Full -> SSE)")
                    response_body = await proxy_response.aread()
                    if prompt_logging_enabled:
                        response_chunks.append(response_body)
                    try:
                        response_json = json.loads(response_body.decode('utf-8'))
                        if 'system_fingerprint' not in response_json:
                            response_json['system_fingerprint'] = system_fingerprint
                        yield f'data: {json.dumps(response_json)}\n\n'.encode('utf-8')
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode non-streaming backend response as JSON. Yielding raw. Status: {proxy_response.status_code}")
                        yield f'data: {response_body.decode("utf-8", errors="replace")}\n\n'.encode('utf-8')
                    yield 'data: [DONE]\n\n'.encode('utf-8')

        except httpx.RequestError as e:
            logging.error(f"Error forwarding stream to runner {model_name}: {e}\n{traceback.format_exc()}")
            error_payload = {"error": {"message": f"Error communicating with runner for model '{model_name}': {e}", "type": "runner_communication_error"}}
            if prompt_logging_enabled:
                 prompts_logger.error(f"Error response from {target_url} for model '{model_name}': {e}")
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return
        except asyncio.TimeoutError as e:
            logging.error(f"Timeout during stream forwarding for {model_name}: {e}\n{traceback.format_exc()}")
            error_payload = {"error": {"message": f"Timeout processing stream for model '{model_name}'.", "type": "request_timeout_error"}}
            if prompt_logging_enabled:
                 prompts_logger.error(f"Timeout processing stream for model '{model_name}': {e}")
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return
        except Exception as e:
            logging.error(f"Unexpected error during stream forwarding for {model_name}: {e}\n{traceback.format_exc()}")
            error_payload = {"error": {"message": f"Internal error processing stream for model '{model_name}': {e}", "type": "internal_error"}}
            if prompt_logging_enabled:
                 prompts_logger.error(f"Unexpected error processing stream for model '{model_name}': {e}")
            yield f'data: {json.dumps(error_payload)}\n\n'.encode('utf-8')
            return
        finally:
            if prompt_logging_enabled and response_chunks:
                 try:
                     full_response_bytes = b''.join(response_chunks)
                     try:
                         full_response_json = json.loads(full_response_bytes.decode('utf-8'))
                         prompts_logger.info(f"Streamed response from {target_url} for model '{model_name}': {json.dumps(full_response_json)}")
                     except json.JSONDecodeError:
                         response_str = full_response_bytes.decode('utf-8', errors='replace')
                         prompts_logger.info(f"Raw streamed response from {target_url} for model '{model_name}': {response_str[:500]}...")
                 except Exception as log_e:
                     logging.error(f"Error logging streamed response body for {request.url.path}: {log_e}")

# --- End of _dynamic_route_v1_request_generator ---


# --- Handlers for /api/v0/* proxying ---
@app.post("/api/v0/chat/completions")
async def _proxy_v0_chat_completions(request: Request):
    """Proxies /api/v0/chat/completions to /v1/chat/completions."""
    # logging.debug(f"Proxying /api/v0/chat/completions to /v1/chat/completions") # F541
    logging.debug("Proxying /api/v0/chat/completions to /v1/chat/completions")
    target_v1_path = "/v1/chat/completions"
    try:
        body_bytes = await request.body()
        body_json = {}
        if body_bytes:
            body_json = json.loads(body_bytes)

        client_requests_stream = body_json.get("stream", False)

        if client_requests_stream:
            return StreamingResponse(content=_dynamic_route_v1_request_generator(
                request,
                target_path=target_v1_path,
                body=body_json,
                body_bytes=body_bytes
            ))
        else:
            response_data = await _fetch_non_streaming_v1_response(
                request,
                target_path=target_v1_path,
                body=body_json,
                body_bytes=body_bytes
            )
            if isinstance(response_data, dict):
                if "error" in response_data:
                    error_type = response_data.get("error", {}).get("type", "unknown_error")
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                    if error_type == "invalid_request_error":
                        status_code = status.HTTP_400_BAD_REQUEST
                    elif error_type == "runner_startup_error" or error_type == "runner_communication_error":
                        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    return JSONResponse(content=response_data, status_code=status_code)
                return JSONResponse(content=response_data)
            else:
                # This should ideally not be reached if _fetch_non_streaming_v1_response adheres to its contract
                logging.error(f"Non-streaming APIv0 request to {target_v1_path} did not return a dict as expected.")
                return JSONResponse(content={"error": {"message": "Internal server error: Invalid response type from processing function.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except json.JSONDecodeError:
        return JSONResponse(content={"error": {"message": "Invalid JSON in request body.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logging.error(f"Error in {target_v1_path} handler: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": {"message": f"Internal server error: {e}", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/api/v0/embeddings")
async def _proxy_v0_embeddings(request: Request):
    """Proxies /api/v0/embeddings to /v1/embeddings. Embeddings are non-streaming."""
    # logging.debug(f"Proxying /api/v0/embeddings to /v1/embeddings") # F541
    logging.debug("Proxying /api/v0/embeddings to /v1/embeddings")
    target_v1_path = "/embeddings"
    try:
        body_bytes = await request.body()
        body_json = {}
        if body_bytes:
            body_json = json.loads(body_bytes)

        # Embeddings are typically non-streaming.
        response_data = await _fetch_non_streaming_v1_response(
            request,
            target_path=target_v1_path,
            body=body_json,
            body_bytes=body_bytes
        )
        if isinstance(response_data, dict):
            if "error" in response_data:
                error_type = response_data.get("error", {}).get("type", "unknown_error")
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                if error_type == "invalid_request_error":
                    status_code = status.HTTP_400_BAD_REQUEST
                elif error_type == "runner_startup_error" or error_type == "runner_communication_error":
                    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                return JSONResponse(content=response_data, status_code=status_code)
            return JSONResponse(content=response_data)
        else:
            logging.error(f"Non-streaming APIv0 request to {target_v1_path} did not return a dict as expected.")
            return JSONResponse(content={"error": {"message": "Internal server error: Invalid response type from processing function.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except json.JSONDecodeError:
        return JSONResponse(content={"error": {"message": "Invalid JSON in request body.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logging.error(f"Error in {target_v1_path} handler: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": {"message": f"Internal server error: {e}", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/api/v0/completions")
async def _proxy_v0_completions(request: Request):
    """Proxies /api/v0/completions to /v1/completions."""
    # logging.debug(f"Proxying /api/v0/completions to /v1/completions") # F541
    logging.debug("Proxying /api/v0/completions to /v1/completions")
    target_v1_path = "/v1/completions"
    try:
        body_bytes = await request.body()
        body_json = {}
        if body_bytes:
            body_json = json.loads(body_bytes)

        client_requests_stream = body_json.get("stream", False)

        if client_requests_stream:
            return StreamingResponse(content=_dynamic_route_v1_request_generator(
                request,
                target_path=target_v1_path,
                body=body_json,
                body_bytes=body_bytes
            ))
        else:
            response_data = await _fetch_non_streaming_v1_response(
                request,
                target_path=target_v1_path,
                body=body_json,
                body_bytes=body_bytes
            )
            if isinstance(response_data, dict):
                if "error" in response_data:
                    error_type = response_data.get("error", {}).get("type", "unknown_error")
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                    if error_type == "invalid_request_error":
                        status_code = status.HTTP_400_BAD_REQUEST
                    elif error_type == "runner_startup_error" or error_type == "runner_communication_error":
                        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    return JSONResponse(content=response_data, status_code=status_code)
                return JSONResponse(content=response_data)
            else:
                logging.error(f"Non-streaming APIv0 request to {target_v1_path} did not return a dict as expected.")
                return JSONResponse(content={"error": {"message": "Internal server error: Invalid response type from processing function.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except json.JSONDecodeError:
        return JSONResponse(content={"error": {"message": "Invalid JSON in request body.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logging.error(f"Error in {target_v1_path} handler: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": {"message": f"Internal server error: {e}", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        body_json = {}
        if body_bytes:
            body_json = json.loads(body_bytes)

        client_requests_stream = body_json.get("stream", False)

        if client_requests_stream:
            return StreamingResponse(content=_dynamic_route_v1_request_generator(
                request, body=body_json, body_bytes=body_bytes
            ))
        else:
            response_data = await _fetch_non_streaming_v1_response(
                request, body=body_json, body_bytes=body_bytes
            )
            if isinstance(response_data, dict):
                if "error" in response_data:
                    error_type = response_data.get("error", {}).get("type", "unknown_error")
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                    if error_type == "invalid_request_error":
                        status_code = status.HTTP_400_BAD_REQUEST
                    elif error_type == "runner_startup_error" or error_type == "runner_communication_error":
                        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    return JSONResponse(content=response_data, status_code=status_code)
                return JSONResponse(content=response_data)
            else:
                logging.error("Non-streaming /v1/chat/completions request did not return a dict as expected.")
                return JSONResponse(content={"error": {"message": "Internal server error: Invalid response type.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except json.JSONDecodeError:
        return JSONResponse(content={"error": {"message": "Invalid JSON in request body.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logging.error(f"Error in /v1/chat/completions handler: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": {"message": f"Internal server error: {e}", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/completions")
async def _v1_completions_handler(request: Request):
    try:
        body_bytes = await request.body()
        body_json = {}
        if body_bytes:
            body_json = json.loads(body_bytes)

        client_requests_stream = body_json.get("stream", False)

        if client_requests_stream:
            return StreamingResponse(content=_dynamic_route_v1_request_generator(
                request, body=body_json, body_bytes=body_bytes
            ))
        else:
            response_data = await _fetch_non_streaming_v1_response(
                request, body=body_json, body_bytes=body_bytes
            )
            if isinstance(response_data, dict):
                if "error" in response_data:
                    error_type = response_data.get("error", {}).get("type", "unknown_error")
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                    if error_type == "invalid_request_error":
                        status_code = status.HTTP_400_BAD_REQUEST
                    elif error_type == "runner_startup_error" or error_type == "runner_communication_error":
                        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    return JSONResponse(content=response_data, status_code=status_code)
                return JSONResponse(content=response_data)
            else:
                logging.error("Non-streaming /v1/completions request did not return a dict as expected.")
                return JSONResponse(content={"error": {"message": "Internal server error: Invalid response type.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": {"message": "Invalid JSON in request body.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logging.error(f"Error in /v1/completions handler: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": {"message": f"Internal server error: {e}", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/embeddings")
async def _v1_embeddings_handler(request: Request):
    try:
        body_bytes = await request.body()
        body_json = {}
        if body_bytes:
            body_json = json.loads(body_bytes)

        # Embeddings are non-streaming.
        response_data = await _fetch_non_streaming_v1_response(
            request, 
            body=body_json, 
            body_bytes=body_bytes,
        )

        if isinstance(response_data, dict):
            if "error" in response_data:
                error_type = response_data.get("error", {}).get("type", "unknown_error")
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                if error_type == "invalid_request_error":
                    status_code = status.HTTP_400_BAD_REQUEST
                elif error_type == "runner_startup_error" or error_type == "runner_communication_error":
                    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                return JSONResponse(content=response_data, status_code=status_code)
            return JSONResponse(content=response_data)
        elif isinstance(response_data, list):
            # If the response is a list, we expect it to be in the LM Studio format
            embedding_obj = response_data[0]
            if isinstance(embedding_obj, dict):
                # Check if it has the expected structure for embeddings
                if 'embedding' in embedding_obj:
                    embedding_arr = embedding_obj.get('embedding')
                    if isinstance(embedding_arr, list) and len(embedding_arr) > 0:
                        # Create one embedding object per vector in the array
                        embedding_objects = [
                            {
                                "object": "embedding",
                                "embedding": vector,
                            }
                            for vector in embedding_arr
                        ]
                        
                        obj_resp = {
                            "object": "list",
                            "data": embedding_objects,
                            "model": body_json.get("model", "unknown_model"),
                            "usage": {
                                "prompt_tokens": 0,
                                "total_tokens": 0
                            }
                        }
                        return JSONResponse(content=obj_resp)
                    else:
                        logging.error("Non-streaming /v1/embeddings request returned an invalid embedding array: " + str(embedding_arr))
                        return JSONResponse(content={"error": {"message": "Invalid embedding data format.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
                else:
                    logging.error("Non-streaming /v1/embeddings request returned an invalid object structure: " + str(embedding_obj))
                    return JSONResponse(content={"error": {"message": "Invalid response structure for embeddings.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                logging.error("Non-streaming /v1/embeddings request returned a list with non-dict items: " + str(response_data))
                return JSONResponse(content={"error": {"message": "Invalid response type for embeddings.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # This case should ideally not be reached if the generator works as expected.
            logging.error("Non-streaming /v1/embeddings request did not return a dict: " + str(response_data) + " of type " + str(type(response_data)))
            return JSONResponse(content={"error": {"message": "Internal server error: Invalid response type from generator for embeddings.", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": {"message": "Invalid JSON in request body.", "type": "invalid_request_error"}}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logging.error(f"Error in /v1/embeddings handler: {e}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": {"message": f"Internal server error: {e}", "type": "internal_error"}}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

logging.info("Updated dynamic routing handlers for /v1/chat/completions, /v1/completions, /v1/embeddings to support conditional streaming.")


# If needed, add a catch-all for other /v1 paths, but be cautious
# @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
# async def _v1_catch_all_handler(request: Request):
#     return await _dynamic_route_v1_request_generator(request)
# logging.info("Added catch-all dynamic routing handler for /v1/*.")


# --- End add routes ---


class LMStudioProxyServer:
    def __init__(self,
                 all_models_config: Dict[str, Any],
                 runtimes_config: Dict[str, Any],
                 is_model_running_callback: Callable[[str], bool],
                 get_runner_port_callback: Callable[[str], Optional[int]],
                 request_runner_start_callback: Callable[[str], asyncio.Future],
                 on_runner_port_ready: Callable[[str, int], None],
                 on_runner_stopped: Callable[[str], None]):
        self.all_models_config = all_models_config
        self.runtimes_config = runtimes_config
        self.is_model_running_callback = is_model_running_callback
        self.get_runner_port_callback = get_runner_port_callback
        self.request_runner_start_callback = request_runner_start_callback
        self._uvicorn_server = None
        self.task = None
        self._runner_ready_futures: Dict[str, asyncio.Future] = {}
        # The callbacks are not used by the server itself but are passed for consistency
        # They will be used by the bridge if needed.

    async def start(self):
        app.state.all_models_config = self.all_models_config
        app.state.runtimes_config = self.runtimes_config
        app.state.is_model_running_callback = self.is_model_running_callback
        app.state.get_runner_port_callback = self.get_runner_port_callback
        app.state.request_runner_start_callback = self.request_runner_start_callback
        app.state.proxy_thread_instance = self  # Maintain compatibility with handlers

        uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=1234, log_level="info")
        self._uvicorn_server = uvicorn.Server(uvicorn_config)

        logging.info("LM Studio Proxy listening on http://127.0.0.1:1234")
        try:
            await self._uvicorn_server.serve()
        except asyncio.CancelledError:
            logging.info("LM Studio Proxy server task cancelled.")
        finally:
            logging.info("LM Studio Proxy server shut down.")

    def stop(self):
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self.task:
            self.task.cancel()