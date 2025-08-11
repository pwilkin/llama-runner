def embeddingRequestFromOllama(ollama_req: dict) -> dict:
    """
    Converts an Ollama embedding request to OpenAI-compatible format.
    Ollama: { model, prompt }
    OpenAI: { model, input }
    """
    return {
        "model": ollama_req.get("model"),
        "input": ollama_req.get("prompt")
    }


def embeddingResponseToOllama(openai_resp: dict) -> dict:
    """
    Converts an OpenAI embedding response to Ollama format.
    OpenAI: { data: [ { embedding, index } ], model, usage }
    Ollama: { embedding }
    """
    first = openai_resp.get("data", [{}])[0]
    return {
        "embedding": first.get("embedding", [])
    }


def generateRequestFromOllama(ollama_req: dict) -> dict:
    """
    Converts an Ollama completion request to OpenAI-compatible format.
    Ollama: { model, prompt, options: { temperature, max_tokens? } }
    OpenAI: { model, prompt, temperature, max_tokens }
    """
    opts = ollama_req.get("options", {})
    return {
        "model": ollama_req.get("model"),
        "prompt": ollama_req.get("prompt"),
        "tools": ollama_req.get("tools"),
        "temperature": opts.get("temperature", 1.0),
        "max_tokens": opts.get("max_tokens")
    }


def generateResponseToOllama(openai_resp: dict) -> dict:
    """
    Converts an OpenAI completion response (including streaming) to Ollama format.
    Handles both full and streaming (delta) responses.
    """
    choice = openai_resp.get("choices", [{}])[0]
    # Handle streaming delta
    delta = choice.get("delta", {})
    response = ""
    if "content" in delta:
        response = delta["content"]
    # Fallback to full text if present
    if not response and "text" in choice:
        response = choice["text"]
    model = openai_resp.get("model", "")
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "model": model,
        "created_at": created_at,
        "response": response,
        "done": choice.get("finish_reason") == "stop"
    }


def chatRequestFromOllama(ollama_req: dict) -> dict:
    """
    Converts an Ollama chat request to OpenAI-compatible format.
    Ollama: { model, messages, options: { temperature, max_tokens? } }
    OpenAI: { model, messages, temperature, max_tokens }
    """
    opts = ollama_req.get("options", {})
    return {
        "model": ollama_req.get("model"),
        "messages": ollama_req.get("messages", []),
        "tools": ollama_req.get("tools"),
        "temperature": opts.get("temperature", 1.0),
        "max_tokens": opts.get("max_tokens")
    }


import datetime

def chatResponseToOllama(openai_resp: dict) -> dict:
    """
    Converts an OpenAI chat completion response (including streaming) to Ollama format.
    Handles both full and streaming (delta) responses.
    """
    choice = openai_resp.get("choices", [{}])[0]
    # Handle streaming delta
    delta = choice.get("delta", {})
    message = {}
    if "role" in delta:
        message["role"] = delta["role"]
    if "content" in delta:
        message["content"] = delta["content"]
    # Fallback to full message if present
    if not message and "message" in choice:
        message = choice["message"]
    # Always include model and created_at
    model = openai_resp.get("model", "")
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "model": model,
        "created_at": created_at,
        "message": message,
        "done": choice.get("finish_reason") == "stop"
    }