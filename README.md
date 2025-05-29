# llama-runner
Llama.cpp runner/swapper and proxy that emulates LMStudio / Ollama backends (for IntelliJ AI Assistant / GitHub Copilot), now with headless mode.

## Installation

```
$ git clone https://github.com/pwilkin/llama-runner
$ cd llama-runner
$ mkdir dev-venv
$ python -m venv dev-venv
$ source dev-venv/bin/activate (or dev-venv\bin\Activate.ps1 on Windows)
$ pip install -r requirements.txt
... create ~/.llama-runner/config.json ...
$ python main.py [--headless]
```

## Sample config file

```json
{
  "llama-runtimes": {
    "default": {
      "runtime": "llama-server"
    },
    "ik_llama": {
      "runtime": "/devel/tools/ik_llama.cpp/build/bin/llama-server",
      "supports_tools": false
    },
    "alt_llama": {
      "runtime": "/devel/tools/other_rel/llama.cpp/build/bin/llama-server"
    }
  },
  "models": {
    "Qwen3 14B": {
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_S.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 26000,
        "gpu_layers": 99,
        "no_kv_offload": true,
        "cache-type-k": "f16",
        "cache-type-v": "q4_0",
        "flash-attn": true,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6,
        "threads": 4,
        "jinja": true
      }
    },
    "Qwen3 14B 128K": {
      "model_id": "Qwen3-14B-128K",
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-14B-128K-GGUF/Qwen3-14B-128K-IQ4_NL.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 80000,
        "gpu_layers": 99,
        "no_kv_offload": true,
        "cache-type-k": "f16",
        "cache-type-v": "q4_0",
        "flash-attn": true,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6,
        "rope-scale": 4,
        "yarn-orig-ctx": 32768,
        "threads": 4,
        "jinja": true
      }
    },
    "Qwen3 8B": {
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q5_K_M.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 25000,
        "gpu_layers": 99,
        "cache-type-k": "f16",
        "cache-type-v": "q4_0",
        "flash-attn": true,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6,
        "threads": 4,
        "jinja": true
      }
    },
    "Qwen3 30B MoE": {
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "override_tensor": "(up_exps|down_exps)=CPU",
        "ctx_size": 40000,
        "flash-attn": true,
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "gpu_layers": 99,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6,
        "threads": 4,
        "jinja": true
      }
    },
    "Qwen3 30B MoE SmallCtx NoRes": {
      "model_id": "Qwen-MoE-smctx-nores",
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "override_tensor": "(up_exps|down_exps)=CPU",
        "ctx_size": 18000,
        "flash-attn": true,
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "gpu_layers": 99,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6,
        "threads": 4,
        "reasoning-budget": 0,
        "jinja": true
      }
    },
    "Qwen2.5 VL": {
      "model_path": "/mnt/win/k/models/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF/Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 8000,
        "gpu_layers": 99,
        "threads": 4,
        "mmproj": "/mnt/win/k/models/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF/mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
        "temp": 0.5
      }
    }, 
    "Seed 8B": {
      "model_path": "/mnt/win/k/models/mradermacher/Seed-Coder-8B-Instruct-i1-GGUF/Seed-Coder-8B-Instruct.i1-Q5_K_S.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "gpu_layers": 99,
        "ctx_size": 30000,
        "flash-attn": true,
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "top_p": 0.8,
        "temp": 0.6,
        "threads": 4
      }
    },
    "Hermes 3B": {
      "model_id": "Hermes-3B",
      "model_path": "/mnt/win/k/models/NousResearch/Hermes-3-Llama-3.2-3B-GGUF/Hermes-3-Llama-3.2-3B.Q5_K_M.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 100000,
        "gpu_layers": 99,
        "cache-type-k": "q4_0",
        "cache-type-v": "q4_0",
        "flash-attn": true,
        "threads": 4
      }
    }
  }
}
```

# Functionality
* headless mode with `--headless` for non-UI systems
* support for different llama.cpp runtimes including ik_llama (for ik_llama, specify "port" in model configuration for runner)
* dynamically loads and unloads runtimes based on model string in request
* dynamically strips tool queries for ik_llama that doesn't support it
* double proxy: emulation for LM Studio-specific backend and OpenAI-compatible backends (running on port 1234) and for Ollama specific backends (running on port 11434)
* tested on GitHub Copilot (for Ollama emulation) and on IntelliJ AI Assistant (for LM Studio emulation)
* tested on Windows & Linux (Ubuntu 24.10)

# Disclaimer

Yes, this is mostly vibe-coded. Pull requests fixing glaring code issues / inefficiencies are welcome. Comments pointing out glaring code issues / inefficiencies are not welcome (unless it's security-critical).
