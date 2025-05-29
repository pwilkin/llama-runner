import asyncio
import logging
import traceback
import json
from typing import Dict, Any, Callable, Optional, AsyncGenerator

import httpx

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from PySide6.QtCore import QThread, Slot

from llama_runner import gguf_metadata
# Removed import: from llama_runner.config_loader import calculate_system_fingerprint
from llama_runner.ollama_proxy_conversions import (
    embeddingRequestFromOllama, embeddingResponseToOllama,
    generateRequestFromOllama, generateResponseToOllama,
    chatRequestFromOllama, chatResponseToOllama
)

# --- Create our own FastAPI app instance ---
app = FastAPI()
# --- End create app instance ---

# Define standalone handlers that access state via app.state

# --- Handler for dynamic routing of requests to Llama.cpp runner ---
async def _dynamic_route_runner_request_generator(request: Request, target_path: str, request_body: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    """
    Intercepts requests, ensures the target runner is running,
    and forwards the request to the runner's port, yielding the response chunks.
    This is an async generator function for streaming responses.
    """
    # Access state and callbacks from the request's app instance
    all_models_config = request.app.state.all_models_config # Use all_models_config
    runtimes_config = request.app.state.runtimes_config # Use runtimes_config
    get_runner_port_callback = request.app.state.get_runner_port_callback
    request_runner_start_callback = request.app.state.request_runner_start_callback
    prompt_logging_enabled = request.app.state.prompt_logging_enabled # Access prompt logging flag
    prompts_logger = request.app.state.prompts_logger # Access prompts logger instance
    # Access the proxy thread instance to get the futures dictionary
    proxy_thread: OllamaProxyThread = request.app.state.proxy_thread_instance

    model_name_from_request = request_body.get("model")
    if not model_name_from_request:
        logging.warning(f"Model name not found in request body for {request.url.path}")
        yield b'data: {"error": "Model name not specified in request body."}\n\n'
        return # Stop the generator

    # Log the incoming request if prompt logging is enabled
    if prompt_logging_enabled:
        try:
            # Log the request body as a JSON string
            prompts_logger.info(f"Request to {request.url.path} for model '{model_name_from_request}': {json.dumps(request_body)}")
        except Exception as log_e:
            logging.error(f"Error logging request body for {request.url.path}: {log_e}")

    logging.debug(f"Intercepted request for model: {model_name_from_request} at path: {request.url.path}")
    logging.debug(f"Available models in all_models_config: {list(all_models_config.keys())}")
    logging.debug(f"Available runtimes in runtimes_config: {list(runtimes_config.keys())}")

    # Check if the model exists in the all_models_config
    if model_name_from_request not in all_models_config:
        logging.warning(f"Request for unknown model: {model_name_from_request}")
        yield f"data: {{\"error\": \"Model '{model_name_from_request}' not found in main configuration.\"}}\n\n".encode('utf-8')
        return # Stop the generator

    # Get the specific model's configuration details from all_models_config
    model_config_details = all_models_config.get(model_name_from_request)
    runtime_name_for_model = model_config_details.get("llama_cpp_runtime") if model_config_details else None

    if not runtime_name_for_model:
        logging.warning(f"Runtime not defined for model '{model_name_from_request}' in main configuration.")
        yield f"data: {{\"error\": \"Runtime not configured for model '{model_name_from_request}'.\"}}\n\n".encode('utf-8')
        return

    # Get the runtime's configuration from runtimes_config
    runtime_details_from_config = runtimes_config.get(runtime_name_for_model)

    if not runtime_details_from_config:
        logging.warning(f"Configuration for runtime '{runtime_name_for_model}' (for model '{model_name_from_request}') not found in runtimes configuration.")
        yield f"data: {{\"error\": \"Configuration for runtime '{runtime_name_for_model}' not found.\"}}\n\n".encode('utf-8')
        return

    # Conditionally remove 'tools' and 'tool_choice' from the request body
    # This is done if the runtime's configuration specifies 'supports_tools': False.
    if runtime_details_from_config.get("supports_tools") is False:
        tools_present_in_request = "tools" in request_body
        tool_choice_present_in_request = "tool_choice" in request_body

        if tools_present_in_request:
            request_body.pop("tools")
        if tool_choice_present_in_request:
            request_body.pop("tool_choice")

        if tools_present_in_request or tool_choice_present_in_request:
            logging.info(
                f"Model '{model_name_from_request}' (using runtime: '{runtime_name_for_model}') "
                f"has supports_tools=False. Removed 'tools' and/or 'tool_choice' from request to {target_path}."
            )
            # request_body is modified in place, and httpx will use the modified version
            # when 'json=request_body' is passed to client.stream()

    # Check if the runner is already running using model_name_from_request
    port = get_runner_port_callback(model_name_from_request)

    if port is None:
        # Runner is not running, request startup and wait
        logging.info(f"Runner for {model_name_from_request} not running. Requesting startup.")
        startup_timeout = 240 # seconds
        try:
            # Request startup via the callback, which returns an asyncio.Future
            # Store the future locally in the proxy thread instance
            if model_name_from_request not in proxy_thread._runner_ready_futures or proxy_thread._runner_ready_futures[model_name_from_request].done():
                 logging.debug(f"Creating new startup future for {model_name_from_request}")
                 proxy_thread._runner_ready_futures[model_name_from_request] = request_runner_start_callback(model_name_from_request)
            else:
                 logging.debug(f"Using existing startup future for {model_name_from_request}")

            # Wait for the runner to become ready (Future to resolve)
            # Use a timeout to prevent infinite waiting
            port = await asyncio.wait_for(
                proxy_thread._runner_ready_futures[model_name_from_request],
                timeout=startup_timeout
            )
            logging.info(f"Runner for {model_name_from_request} is ready on port {port} after startup.")

        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for runner {model_name_from_request} to start after {startup_timeout} seconds.")
            # Clean up the future if it timed out
            if model_name_from_request in proxy_thread._runner_ready_futures and not proxy_thread._runner_ready_futures[model_name_from_request].done():
                 proxy_thread._runner_ready_futures[model_name_from_request].cancel() # Cancel the future
                 del proxy_thread._runner_ready_futures[model_name_from_request]
            yield f"data: {{\"error\": \"Timeout starting runner for model '{model_name_from_request}'.\"}}\n\n".encode('utf-8')
            return # Stop the generator
        except Exception as e:
            logging.error(f"Error during runner startup for {model_name_from_request}: {e}\n{traceback.format_exc()}")
            # Clean up the future if it failed
            if model_name_from_request in proxy_thread._runner_ready_futures and not proxy_thread._runner_ready_futures[model_name_from_request].done():
                 proxy_thread._runner_ready_futures[model_name_from_request].set_exception(e) # Set exception on the future
                 del proxy_thread._runner_ready_futures[model_name_from_request]
            yield f"data: {{\"error\": \"Error starting runner for model '{model_name_from_request}': {e}\"}}\n\n".encode('utf-8')
            return # Stop the generator

    else:
        logging.debug(f"Runner for {model_name_from_request} is already running on port {port}.")
        # If it was running, ensure its future is marked as done with the port
        # This handles cases where the proxy restarts but the runner is still alive
        if model_name_from_request not in proxy_thread._runner_ready_futures or not proxy_thread._runner_ready_futures[model_name_from_request].done():
             logging.debug(f"Creating completed future for already running runner {model_name_from_request}")
             future = asyncio.Future()
             future.set_result(port)
             proxy_thread._runner_ready_futures[model_name_from_request] = future
             # No need for timer cleanup here, as it's already running.
             # The future will be removed if the runner stops later.


    # Runner is ready and port is known. Forward the request.
    target_url = f"http://127.0.0.1:{port}{target_path}"
    logging.debug(f"Target URL: {target_url}")
    logging.debug(f"Forwarding request for {model_name_from_request} to {target_url}")

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
            # Use the original request method and the converted request body
            async with client.stream(
                method=request.method,
                url=target_url,
                headers=headers,
                json=request_body, # Pass the converted JSON body
                timeout=600.0, # Use a generous timeout for model responses
            ) as proxy_response:

                # Check if the response is streaming (e.g., Server-Sent Events)
                content_type = proxy_response.headers.get('content-type', '').lower()
                is_streaming = 'text/event-stream' in content_type

                if is_streaming:
                    logging.debug(f"Forwarding streaming response from {target_url}")
                    # Yield chunks directly from the httpx stream within this generator
                    async for chunk in proxy_response.aiter_bytes():
                        if prompt_logging_enabled:
                             response_chunks.append(chunk) # Buffer chunk for logging
                        yield chunk # Yield each chunk as it arrives
                else:
                    logging.debug(f"Forwarding non-streaming response from {target_url}")
                    # For non-streaming responses, read the body and yield it as a single chunk
                    response_body = await proxy_response.aread() # Read the entire body asynchronously
                    if prompt_logging_enabled:
                         response_chunks.append(response_body) # Buffer the full body
                    yield response_body # Yield the original body as one chunk

        except httpx.RequestError as e:
            logging.error(f"Error forwarding request to runner {model_name_from_request} on port {port}: {e}\n{traceback.format_exc()}")
            # Log the error response if prompt logging is enabled
            if prompt_logging_enabled:
                 prompts_logger.error(f"Error response from {target_url} for model '{model_name_from_request}': {e}")
            yield f'data: {{"error": "Error communicating with runner for model \'{model_name_from_request}\': {e}"}}\n\n'.encode('utf-8')

        except Exception as e:
            logging.error(f"Unexpected error during request forwarding for {model_name_from_request}: {e}\n{traceback.format_exc()}")
            # Log the unexpected error if prompt logging is enabled
            if prompt_logging_enabled:
                 prompts_logger.error(f"Unexpected error processing request for model '{model_name_from_request}': {e}")
            yield f'data: {{"error": "Internal error processing request for model \'{model_name_from_request}\': {e}"}}\n\n'.encode('utf-8')

        finally:
            # Log the complete response if prompt logging is enabled and no major error occurred during forwarding
            if prompt_logging_enabled and response_chunks:
                 try:
                     full_response_bytes = b''.join(response_chunks)
                     # Attempt to decode and log as JSON if possible, otherwise log raw bytes
                     try:
                         full_response_json = json.loads(full_response_bytes.decode('utf-8'))
                         prompts_logger.info(f"Response from {target_url} for model '{model_name_from_request}': {json.dumps(full_response_json)}")
                     except json.JSONDecodeError:
                         # If not JSON, log the raw string or a truncated version
                         response_str = full_response_bytes.decode('utf-8', errors='replace')
                         prompts_logger.info(f"Raw response from {target_url} for model '{model_name_from_request}': {response_str[:500]}...") # Log first 500 chars
                 except Exception as log_e:
                     logging.error(f"Error logging response body for {request.url.path}: {log_e}")

# --- End handler for dynamic routing ---


# --- Handlers for Ollama API endpoints ---

@app.post("/api/generate")
async def generate_completion(request: Request):
    """Handles Ollama /api/generate requests."""
    try:
        ollama_req = await request.json()
        openai_req = generateRequestFromOllama(ollama_req)
        # Ensure the stream parameter is passed through to the OpenAI backend
        openai_req["stream"] = ollama_req.get("stream", True)
        is_stream = ollama_req.get("stream", True)

        if not is_stream:
            # Non-streaming: forward, read full response, convert, return JSONResponse
            async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/completions", request_body=openai_req):
                try:
                    openai_resp = json.loads(chunk.decode('utf-8'))
                    ollama_resp = generateResponseToOllama(openai_resp)
                    ollama_resp["done"] = True
                    return JSONResponse(content=ollama_resp)
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode JSON from runner response: {chunk.decode('utf-8')}")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid JSON response from runner")
                except Exception as conv_e:
                    logging.error(f"Error converting OpenAI response to Ollama format: {conv_e}\n{traceback.format_exc()}")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error converting response")
            # If the generator finishes without yielding, it means an error occurred in the router
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error routing request to runner")
        else:
            # Streaming: use the generator logic
            import time
            async def generate_response_stream():
                prev_ollama_resp = None
                saw_stop = False
                start_time_ns = time.perf_counter_ns()
                eval_count = 0
                first_token_time_ns = None
                last_token_time_ns = None

                async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/completions", request_body=openai_req):
                    try:
                        logging.debug(f"Received chunk from backend (len={len(chunk)}): {repr(chunk[:200])}")
                        chunk_str = chunk.decode('utf-8').strip()
                        logging.debug(f"Decoded chunk_str: {repr(chunk_str[:200])}")
                        # Try to handle both SSE (data: ...) and plain JSON
                        blocks = []
                        if "data: " in chunk_str:
                            blocks = [b[len("data: "):] for b in chunk_str.split('\n\n') if b.startswith("data: ")]
                            logging.debug(f"SSE blocks found: {len(blocks)}")
                        elif chunk_str:
                            blocks = [chunk_str]
                            logging.debug("No SSE prefix, treating as single JSON block.")
                        else:
                            logging.debug("Empty chunk_str after decode/strip.")
                        for json_payload in blocks:
                            block_stripped = json_payload.strip()
                            # Extra debug for [DONE] handling
                            logging.debug(f"Block raw: {repr(json_payload)} | stripped: {repr(block_stripped)}")
                            # Robust skip for [DONE] marker (ignore case, whitespace, CRLF)
                            if not block_stripped or block_stripped.upper().replace('\r', '').replace('\n', '') == "[DONE]":
                                logging.debug(f"Skipping block: {repr(json_payload[:200])}")
                                continue
                            # Only log and process if not skipped
                            logging.debug(f"Block raw: {repr(json_payload)} | stripped: {repr(block_stripped)}")
                            if not block_stripped or block_stripped.upper().replace('\r', '').replace('\n', '') == "[DONE]":
                                logging.debug(f"Skipping block: {repr(json_payload[:200])}")
                                continue
                            logging.debug(f"Processing block: {repr(json_payload[:200])}")
                            try:
                                openai_resp = json.loads(json_payload)
                                logging.debug(f"Parsed OpenAI JSON: {openai_resp}")
                                ollama_resp = generateResponseToOllama(openai_resp)
                                logging.debug(f"Converted to Ollama format: {ollama_resp}")
                                # Determine if this is the final chunk (OpenAI: finish_reason == "stop")
                                finish_reason = None
                                if isinstance(openai_resp, dict):
                                    if "finish_reason" in openai_resp:
                                        finish_reason = openai_resp.get("finish_reason")
                                    elif "choices" in openai_resp and openai_resp["choices"]:
                                        finish_reason = openai_resp["choices"][0].get("finish_reason")
                                if prev_ollama_resp is not None:
                                    prev_ollama_resp["done"] = False
                                    logging.debug(f"Yielding Ollama chunk (done: false): {prev_ollama_resp}")
                                    resp_to_yield = prev_ollama_resp
                                    if not isinstance(resp_to_yield, dict):
                                        resp_to_yield = {"response": str(resp_to_yield)}
                                    # Count eval tokens (non-empty content)
                                    if resp_to_yield.get("response"):
                                        eval_count += 1
                                        now_ns = time.perf_counter_ns()
                                        if first_token_time_ns is None:
                                            first_token_time_ns = now_ns
                                        last_token_time_ns = now_ns
                                    yield (json.dumps(resp_to_yield) + "\n").encode('utf-8')
                                prev_ollama_resp = ollama_resp
                                if finish_reason == "stop":
                                    prev_ollama_resp["done"] = True
                                    logging.debug(f"Yielding Ollama chunk (done: true): {prev_ollama_resp}")
                                    resp_to_yield = prev_ollama_resp
                                    if not isinstance(resp_to_yield, dict):
                                        resp_to_yield = {"response": str(resp_to_yield)}
                                    # Add Ollama-style final fields
                                    end_time_ns = time.perf_counter_ns()
                                    resp_to_yield["done_reason"] = "stop"
                                    resp_to_yield["total_duration"] = str(end_time_ns - start_time_ns)
                                    resp_to_yield["load_duration"] = "0"
                                    resp_to_yield["prompt_eval_count"] = "0"
                                    resp_to_yield["prompt_eval_duration"] = "0"
                                    resp_to_yield["eval_count"] = str(eval_count)
                                    if first_token_time_ns is not None and last_token_time_ns is not None:
                                        resp_to_yield["eval_duration"] = str(last_token_time_ns - first_token_time_ns)
                                    else:
                                        resp_to_yield["eval_duration"] = "0"
                                    yield (json.dumps(resp_to_yield) + "\n").encode('utf-8')
                                    prev_ollama_resp = None
                                    saw_stop = True
                                    break
                            except json.JSONDecodeError:
                                logging.warning(f"Could not decode JSON from runner response chunk: {json_payload}")
                                yield f'data: {{"error": "Invalid JSON from runner: {json_payload}"}}\n\n'.encode('utf-8')
                            except Exception as conv_e:
                                logging.error(f"Error converting OpenAI response to Ollama format: {conv_e}\n{traceback.format_exc()}")
                                yield f'data: {{"error": "Error converting response: {conv_e}"}}\n\n'.encode('utf-8')
                    except Exception as e:
                        logging.error(f"Error processing runner response chunk: {e}\n{traceback.format_exc()}")
                        yield f'data: {{"error": "Error processing response chunk: {e}"}}\n\n'.encode('utf-8')

                # If stream ended and we have a buffered chunk, yield it as done: true
                if prev_ollama_resp is not None and not saw_stop:
                    prev_ollama_resp["done"] = True
                    # Remove empty message/response fields for final chunk
                    if "message" in prev_ollama_resp and (not prev_ollama_resp["message"] or prev_ollama_resp["message"] == {}):
                        del prev_ollama_resp["message"]
                    if "response" in prev_ollama_resp and (not prev_ollama_resp["response"] or prev_ollama_resp["response"] == ""):
                        del prev_ollama_resp["response"]
                    resp_to_yield = prev_ollama_resp
                    if not isinstance(resp_to_yield, dict):
                        resp_to_yield = {"response": str(resp_to_yield)}
                    # Add Ollama-style final fields
                    end_time_ns = time.perf_counter_ns()
                    resp_to_yield["done_reason"] = "stop"
                    resp_to_yield["total_duration"] = str(end_time_ns - start_time_ns)
                    resp_to_yield["load_duration"] = "0"
                    resp_to_yield["prompt_eval_count"] = "0"
                    resp_to_yield["prompt_eval_duration"] = "0"
                    resp_to_yield["eval_count"] = str(eval_count)
                    if first_token_time_ns is not None and last_token_time_ns is not None:
                        resp_to_yield["eval_duration"] = str(last_token_time_ns - first_token_time_ns)
                    else:
                        resp_to_yield["eval_duration"] = "0"
                    yield (json.dumps(resp_to_yield) + "\n").encode('utf-8')

            return StreamingResponse(content=generate_response_stream(), media_type="text/event-stream")

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except Exception as e:
        logging.error(f"Error handling /api/generate: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing generate request")


@app.post("/api/chat")
async def chat_completion(request: Request):
    """Handles Ollama /api/chat requests."""
    try:
        ollama_req = await request.json()
        openai_req = chatRequestFromOllama(ollama_req)
        # Ensure the stream parameter is passed through to the OpenAI backend
        openai_req["stream"] = ollama_req.get("stream", True)
        is_stream = ollama_req.get("stream", True)

        if not is_stream:
            # Non-streaming: forward, read full response, convert, return JSONResponse
            async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/chat/completions", request_body=openai_req):
                try:
                    openai_resp = json.loads(chunk.decode('utf-8'))
                    ollama_resp = chatResponseToOllama(openai_resp)
                    ollama_resp["done"] = True
                    return JSONResponse(content=ollama_resp)
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode JSON from runner response: {chunk.decode('utf-8')}")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid JSON response from runner")
                except Exception as conv_e:
                    logging.error(f"Error converting OpenAI response to Ollama format: {conv_e}\n{traceback.format_exc()}")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error converting response")
            # If the generator finishes without yielding, it means an error occurred in the router
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error routing request to runner")
        else:
            # Streaming: use the generator logic
            import time
            async def chat_response_stream():
                prev_ollama_resp = None
                saw_stop = False
                start_time_ns = time.perf_counter_ns()
                eval_count = 0
                first_token_time_ns = None
                last_token_time_ns = None

                async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/chat/completions", request_body=openai_req):
                    try:
                        logging.debug(f"Received chunk from backend (len={len(chunk)}): {repr(chunk[:200])}")
                        chunk_str = chunk.decode('utf-8').strip()
                        logging.debug(f"Decoded chunk_str: {repr(chunk_str[:200])}")
                        blocks = []
                        if "data: " in chunk_str:
                            blocks = [b[len("data: "):] for b in chunk_str.split('\n\n') if b.startswith("data: ")]
                            logging.debug(f"SSE blocks found: {len(blocks)}")
                        elif chunk_str:
                            blocks = [chunk_str]
                            logging.debug("No SSE prefix, treating as single JSON block.")
                        else:
                            logging.debug("Empty chunk_str after decode/strip.")
                        for json_payload in blocks:
                            # Only log and process if not skipped
                            block_stripped = json_payload.strip()
                            logging.debug(f"Block raw: {repr(json_payload)} | stripped: {repr(block_stripped)}")
                            if not block_stripped or block_stripped.upper().replace('\r', '').replace('\n', '') == "[DONE]":
                                logging.debug(f"Skipping block: {repr(json_payload[:200])}")
                                continue
                            logging.debug(f"Processing block: {repr(json_payload[:200])}")
                            try:
                                openai_resp = json.loads(json_payload)
                                logging.debug(f"Parsed OpenAI JSON: {openai_resp}")
                                ollama_resp = chatResponseToOllama(openai_resp)
                                logging.debug(f"Converted to Ollama format: {ollama_resp}")
                                finish_reason = None
                                if isinstance(openai_resp, dict):
                                    if "finish_reason" in openai_resp:
                                        finish_reason = openai_resp.get("finish_reason")
                                    elif "choices" in openai_resp and openai_resp["choices"]:
                                        finish_reason = openai_resp["choices"][0].get("finish_reason")
                                if prev_ollama_resp is not None:
                                    prev_ollama_resp["done"] = False
                                    logging.debug(f"Yielding Ollama chunk (done: false): {prev_ollama_resp}")
                                    resp_to_yield = prev_ollama_resp
                                    if not isinstance(resp_to_yield, dict):
                                        resp_to_yield = {"message": str(resp_to_yield)}
                                    # Count eval tokens (non-empty content)
                                    msg = resp_to_yield.get("message", {})
                                    if isinstance(msg, dict) and msg.get("content"):
                                        eval_count += 1
                                        now_ns = time.perf_counter_ns()
                                        if first_token_time_ns is None:
                                            first_token_time_ns = now_ns
                                        last_token_time_ns = now_ns
                                    yield (json.dumps(resp_to_yield) + "\n").encode('utf-8')
                                prev_ollama_resp = ollama_resp
                                if finish_reason == "stop":
                                    prev_ollama_resp["done"] = True
                                    logging.debug(f"Yielding Ollama chunk (done: true): {prev_ollama_resp}")
                                    resp_to_yield = prev_ollama_resp
                                    if not isinstance(resp_to_yield, dict):
                                        resp_to_yield = {"message": str(resp_to_yield)}
                                    # Add Ollama-style final fields
                                    end_time_ns = time.perf_counter_ns()
                                    resp_to_yield["done_reason"] = "stop"
                                    resp_to_yield["total_duration"] = str(end_time_ns - start_time_ns)
                                    resp_to_yield["load_duration"] = "0"
                                    resp_to_yield["prompt_eval_count"] = "0"
                                    resp_to_yield["prompt_eval_duration"] = "0"
                                    resp_to_yield["eval_count"] = str(eval_count)
                                    if first_token_time_ns is not None and last_token_time_ns is not None:
                                        resp_to_yield["eval_duration"] = str(last_token_time_ns - first_token_time_ns)
                                    else:
                                        resp_to_yield["eval_duration"] = "0"
                                    yield (json.dumps(resp_to_yield) + "\n").encode('utf-8')
                                    prev_ollama_resp = None
                                    saw_stop = True
                                    break
                            except json.JSONDecodeError:
                                logging.warning(f"Could not decode JSON from runner response chunk: {json_payload}")
                                yield f'data: {{"error": "Invalid JSON from runner: {json_payload}"}}\n\n'.encode('utf-8')
                            except Exception as conv_e:
                                logging.error(f"Error converting OpenAI response to Ollama format: {conv_e}\n{traceback.format_exc()}")
                                yield f'data: {{"error": "Error converting response: {conv_e}"}}\n\n'.encode('utf-8')
                    except Exception as e:
                        logging.error(f"Error processing runner response chunk: {e}\n{traceback.format_exc()}")
                        yield f'data: {{"error": "Error processing response chunk: {e}"}}\n\n'.encode('utf-8')

                if prev_ollama_resp is not None and not saw_stop:
                    prev_ollama_resp["done"] = True
                    # Remove empty message/response fields for final chunk
                    if "message" in prev_ollama_resp and (not prev_ollama_resp["message"] or prev_ollama_resp["message"] == {}):
                        del prev_ollama_resp["message"]
                    if "response" in prev_ollama_resp and (not prev_ollama_resp["response"] or prev_ollama_resp["response"] == ""):
                        del prev_ollama_resp["response"]
                    resp_to_yield = prev_ollama_resp
                    if not isinstance(resp_to_yield, dict):
                        resp_to_yield = {"message": str(resp_to_yield)}
                    # Add Ollama-style final fields
                    end_time_ns = time.perf_counter_ns()
                    resp_to_yield["done_reason"] = "stop"
                    resp_to_yield["total_duration"] = str(end_time_ns - start_time_ns)
                    resp_to_yield["load_duration"] = "0"
                    resp_to_yield["prompt_eval_count"] = "0"
                    resp_to_yield["prompt_eval_duration"] = "0"
                    resp_to_yield["eval_count"] = str(eval_count)
                    if first_token_time_ns is not None and last_token_time_ns is not None:
                        resp_to_yield["eval_duration"] = str(last_token_time_ns - first_token_time_ns)
                    else:
                        resp_to_yield["eval_duration"] = "0"
                    yield (json.dumps(resp_to_yield) + "\n").encode('utf-8')

            return StreamingResponse(content=chat_response_stream(), media_type="text/event-stream")

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except Exception as e:
        logging.error(f"Error handling /api/chat: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing chat request")


@app.post("/api/embeddings")
async def generate_embeddings(request: Request):
    """Handles Ollama /api/embeddings requests."""
    try:
        ollama_req = await request.json()
        openai_req = embeddingRequestFromOllama(ollama_req)

        # Use the dynamic router to forward the converted request to the runner
        # The target path for embeddings in OpenAI format is typically /v1/embeddings
        # Embeddings are typically non-streaming
        async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/embeddings", request_body=openai_req):
            # For non-streaming, the generator yields the entire body as one chunk
            try:
                openai_resp = json.loads(chunk.decode('utf-8'))
                ollama_resp = embeddingResponseToOllama(openai_resp)
                return JSONResponse(content=ollama_resp)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from runner response: {chunk.decode('utf-8')}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid JSON response from runner")
            except Exception as conv_e:
                logging.error(f"Error converting OpenAI response to Ollama format: {conv_e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error converting response")

        # If the generator finishes without yielding, it means an error occurred in the router
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error routing request to runner")

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except Exception as e:
        logging.error(f"Error handling /api/embeddings: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing embeddings request")


@app.get("/api/tags")
async def list_models(request: Request):
    """Handles Ollama /api/tags requests."""
    try:
        # Access the main models configuration from app.state
        all_models_config = request.app.state.all_models_config
        model_list = []
        for name, config in all_models_config.items(): # Iterate over all_models_config
            # Construct the model dictionary based on the expected format
            # Assuming config contains 'modified_at', 'size', 'digest', and 'details'
            model_entry = {
                "name": name,
                "modified_at": config.get("modified_at", ""), # Provide default empty string if key is missing
                "size": config.get("size", 0), # Provide default 0 if key is missing
                "digest": config.get("digest", ""), # Provide default empty string if key is missing
                "details": config.get("details", {}) # Provide default empty dict if key is missing
            }
            model_list.append(model_entry)

        return JSONResponse(content={"models": model_list})

    except Exception as e:
        logging.error(f"Error handling /api/tags: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing list models request")



# --- End handlers for Ollama API endpoints ---

@app.post("/api/show")
async def show_model_info(request: Request):
    """Handles Ollama /api/show requests."""
    try:
        request_body = await request.json()
        model_name = request_body.get("model")

        if not model_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model name not specified in request body")

        models_config = request.app.state.all_models_config # Corrected state variable name

        if model_name not in models_config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' not found")

        model_config = models_config[model_name]
        model_path = model_config.get("model_path")

        if not model_path:
             # This shouldn't happen if models_config is correctly populated, but handle defensively
             logging.error(f"Model config for {model_name} is missing 'path'. Config: {model_config}")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Configuration error for model '{model_name}'")

        # Access or load GGUF metadata
        try:
            # Use get_model_lmstudio_format to get metadata, including running state
            is_running = request.app.state.proxy_thread_instance.is_model_running_callback(model_name)
            model_config = request.app.state.all_models_config.get(model_name)
            metadata = gguf_metadata.get_model_lmstudio_format(model_name, model_path, model_config, is_running)

            if not metadata:
                 logging.error(f"Failed to get metadata for {model_name} at {model_path}")
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get metadata for model '{model_name}'")

            chat_template = metadata.get("raw_metadata", {}).get("chat_template", "") # Get chat_template from raw_metadata
            model_info = metadata # Use the full LM Studio format metadata
        except Exception as e:
            logging.error(f"Error accessing GGUF metadata for {model_name} at {model_path}: {e}\n{traceback.format_exc()}")
            # Decide whether to raise an error or return partial info.
            # For now, raise an error as metadata is a required part of the response.
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error accessing metadata for model '{model_name}'")

        # Construct the 'details' object
        details = {
            "parent_model": model_config.get("parent_model", ""),
            "format": model_config.get("format", ""),
            "family": model_config.get("details", {}).get("architecture", ""), # Map architecture to family
            "families": model_config.get("families", []),
            "parameter_size": model_config.get("parameter_size", ""),
            "quantization_level": model_config.get("quantization_level", ""),
        }

        # Construct the final response
        response_content = {
            "modelfile": "",
            "parameters": "",
            "template": chat_template,
            "details": details,
            "model_info": model_info,
            "capabilities": ["completion", "vision"], # Hardcoded capabilities
        }

        return JSONResponse(content=response_content)

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except HTTPException as http_exc:
        # Re-raise explicit HTTPExceptions
        raise http_exc
    except Exception as e:
        logging.error(f"Error handling /api/show: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing show request")

# --- End handlers for Ollama API endpoints ---

# --- Handlers for OpenAI compatible API endpoints (v1) ---

@app.get("/v1/models")
async def list_openai_models(request: Request):
    """Handles OpenAI /v1/models requests."""
    try:
        models_config = request.app.state.all_models_config # Corrected state variable name
        model_list = []
        # Placeholder values for fields not available in models_config
        created_placeholder = 1678880000 # Example timestamp
        owned_by_placeholder = "llama-runner"

        for name, config in models_config.items():
            # Construct the model dictionary based on the expected OpenAI format
            model_entry = {
                "id": name,
                "object": "model",
                "created": created_placeholder,
                "owned_by": owned_by_placeholder,
                # Add other relevant fields if available in models_config and needed
                # "size": config.get("size", 0),
                # "details": config.get("details", {})
            }
            model_list.append(model_entry)

        return JSONResponse(content={"object": "list", "data": model_list})

    except Exception as e:
        logging.error(f"Error handling /v1/models: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing list models request")

# --- End handlers for OpenAI compatible API endpoints (v1) ---

@app.post("/v1/completions")
async def openai_completions(request: Request):
    """Handles OpenAI /v1/completions requests."""
    try:
        request_body = await request.json()
        # No conversion needed here, assuming the incoming request is already OpenAI format
        # If conversion from a different format (e.g., Ollama) was needed, it would happen here.
        # For this task, we assume the runner expects OpenAI format directly on /v1 paths.

        async def completion_response_stream():
            # The dynamic router handles the forwarding and streaming
            async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/completions", request_body=request_body):
                yield chunk # Yield chunks directly from the runner

        # Return StreamingResponse with the generator
        return StreamingResponse(content=completion_response_stream(), media_type="text/event-stream")

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except Exception as e:
        logging.error(f"Error handling /v1/completions: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing completions request")

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """Handles OpenAI /v1/chat/completions requests."""
    try:
        request_body = await request.json()
        # No conversion needed here, assuming the incoming request is already OpenAI format

        async def chat_completion_response_stream():
            # The dynamic router handles the forwarding and streaming
            async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/chat/completions", request_body=request_body):
                yield chunk # Yield chunks directly from the runner

        # Return StreamingResponse with the generator
        return StreamingResponse(content=chat_completion_response_stream(), media_type="text/event-stream")

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except Exception as e:
        logging.error(f"Error handling /v1/chat/completions: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing chat completions request")

@app.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    """Handles OpenAI /v1/embeddings requests."""
    try:
        request_body = await request.json()
        # No conversion needed here, assuming the incoming request is already OpenAI format

        # Embeddings are typically non-streaming
        async for chunk in _dynamic_route_runner_request_generator(request, target_path="/v1/embeddings", request_body=request_body):
            # For non-streaming, the generator yields the entire body as one chunk
            try:
                # Assuming the runner returns JSON directly for non-streaming
                response_content = json.loads(chunk.decode('utf-8'))
                return JSONResponse(content=response_content)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from runner response for embeddings: {chunk.decode('utf-8')}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid JSON response from runner for embeddings")
            except Exception as e:
                logging.error(f"Error processing runner response for embeddings: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing embeddings response")

        # If the generator finishes without yielding, it means an error occurred in the router
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error routing request to runner for embeddings")

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body")
    except Exception as e:
        logging.error(f"Error handling /v1/embeddings: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error processing embeddings request")


class OllamaProxyThread(QThread):
    """
    QThread to run the FastAPI proxy emulating the Ollama API in a separate thread.
    """
    # Define signals if needed (e.g., started, stopped, error)
    # started = Signal()
    # stopped = Signal()
    # error = Signal(str)

    def __init__(self,
                 all_models_config: Dict[str, Dict[str, Any]], # Renamed models_config to all_models_config
                 runtimes_config: Dict[str, Dict[str, Any]], # Renamed models_config to runtimes_config
                 is_model_running_callback: Callable[[str], bool],
                 get_runner_port_callback: Callable[[str], Optional[int]],
                 request_runner_start_callback: Callable[[str], asyncio.Future],
                 prompt_logging_enabled: bool, # Add prompt logging flag
                 prompts_logger: logging.Logger): # Add prompts logger instance
        super().__init__()
        self.all_models_config = all_models_config # Store all_models_config
        self.runtimes_config = runtimes_config # Store runtimes_config
        self.is_model_running_callback = is_model_running_callback
        self.get_runner_port_callback = get_runner_port_callback
        self.request_runner_start_callback = request_runner_start_callback
        self.prompt_logging_enabled = prompt_logging_enabled # Store the flag
        self.prompts_logger = prompts_logger # Store the logger instance
        self.is_running = False
        self._uvicorn_server = None

        # Dictionary to hold asyncio Futures for runners that are starting
        self._runner_ready_futures: Dict[str, asyncio.Future] = {}

        # Connect to signals from MainWindow (MainWindow connects these signals to our slots)
        # Signals: runner_port_ready_for_proxy, runner_stopped_for_proxy
        # Slots: on_runner_port_ready, on_runner_stopped


    @Slot(str, int)
    def on_runner_port_ready(self, model_name: str, port: int):
        """Slot to handle runner_port_ready_for_proxy signal from MainWindow."""
        logging.debug(f"Ollama Proxy thread received runner_port_ready for {model_name} on port {port}")
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
        logging.debug(f"Ollama Proxy thread received runner_stopped for {model_name}")
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
            logging.error(f"Unexpected error in OllamaProxyThread run: {e}\n{traceback.format_exc()}")
        finally:
            self.is_running = False

            # Log proxy stop using the prompts logger if enabled
            if self.prompt_logging_enabled:
                 self.prompts_logger.info("Ollama Proxy thread stopping.")

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
        print("Starting Ollama Proxy...")
        try:
            # Set state on the global app instance BEFORE creating the server
            # This state will be accessible by the standalone handler functions
            app.state.all_models_config = self.all_models_config # Pass all_models_config
            app.state.runtimes_config = self.runtimes_config # Pass runtimes_config
            app.state.is_model_running_callback = self.is_model_running_callback
            app.state.get_runner_port_callback = self.get_runner_port_callback
            app.state.request_runner_start_callback = self.request_runner_start_callback
            app.state.prompt_logging_enabled = self.prompt_logging_enabled # Set prompt logging flag on state
            app.state.prompts_logger = self.prompts_logger # Set prompts logger on state

            # Use port 11434 as required for Ollama emulation
            uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=11434, reload=False)
            self._uvicorn_server = uvicorn.Server(uvicorn_config)

            print("Ollama Proxy listening on http://127.0.0.1:11434")
            logging.info("Ollama Proxy listening on http://127.0.0.1:11434")

            # Log proxy start using the prompts logger if enabled
            if self.prompt_logging_enabled:
                 self.prompts_logger.info("Ollama Proxy thread started and listening on http://127.0.0.1:11434")


            # This call is blocking until the server stops
            await self._uvicorn_server.serve()

        except Exception as e:
            print(f"Error starting Ollama Proxy: {e}")
            logging.error(f"Error starting Ollama Proxy: {e}\n{traceback.format_exc()}")
            # Emit error signal if needed
            # self.error.emit(str(e))
        finally:
            # Cleanup happens in run()'s finally block now
            print("Ollama Proxy stopped.")
            logging.info("Ollama Proxy stopped.")


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

