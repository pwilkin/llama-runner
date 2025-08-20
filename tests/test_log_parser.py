#!/usr/bin/env python3
"""
Test script for the log parser functionality.
This script tests the LlamaLogParser with sample log data.
"""

import sys
import os

# Add the project root to the path so we can import llama_runner modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_log_parser():
    """Test the log parser with sample log data."""
    
    # Sample log data from llama.cpp server
    sample_logs = [
        "llama_model_loader: loaded meta data with 23 key-value pairs and 291 tensors from /Users/user/models/llama-2-7b-chat.Q4_K_M.gguf (version GGUF V3 (latest))",
        "llm_load_print_meta: format           = GGUF V3 (latest)",
        "llm_load_print_meta: arch             = llama",
        "llm_load_print_meta: vocab type       = SPM",
        "llm_load_print_meta: n_vocab          = 32000",
        "llm_load_print_meta: n_merges         = 0",
        "llm_load_print_meta: n_ctx_train      = 4096",
        "llm_load_print_meta: n_embd           = 4096",
        "llm_load_print_meta: n_head           = 32",
        "llm_load_print_meta: n_layer          = 32",
        "llm_load_print_meta: n_rot            = 128",
        "llm_load_print_meta: n_gqa            = 1",
        "llama_new_context_with_model: n_ctx      = 4096",
        "llama_new_context_with_model: n_batch    = 512",
        "llama_new_context_with_model: n_ubatch   = 512",
        "llama_new_context_with_model: flash_attn = 0",
        "llama_new_context_with_model: freq_base  = 10000.0",
        "llama_new_context_with_model: freq_scale = 1",
        "llama_kv_cache_init:  CUDA_Host  KV buffer size = 1024.00 MiB",
        "llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB",
        "llama_new_context_with_model:  CUDA_Host  output buffer size =     0.12 MiB",
        "llama_new_context_with_model:      CUDA0 compute buffer size =   264.00 MiB",
        "llama_new_context_with_model:  CUDA_Host compute buffer size =    16.01 MiB",
        "llama_new_context_with_model: graph nodes  = 1030",
        "llama_new_context_with_model: graph splits = 2",
        "llama_model_loader: - type  f32:   65 tensors",
        "llama_model_loader: - type q4_K:  193 tensors",
        "llama_model_loader: - type q6_K:   33 tensors",
        "srv  update_slots: all slots are idle",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "request: POST /completion 127.0.0.1 200",
    ]

    parser = LlamaLogParser()
    status_info = parser.parse_multiple_lines(sample_logs)
    
    # We expect IDLE status since there's no timing information in this log
    assert status_info.status == ModelStatus.IDLE, f"Expected IDLE status, got {status_info.status}"

def test_timing_log_parser():
    """Test the log parser with timing information in logs."""
    timing_logs = [
        "main: server is listening on http://127.0.0.1:40689 - starting the main loop",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: GET / 127.0.0.1 200",
        "srv  log_server_r: request: GET /props 127.0.0.1 200",
        "srv  log_server_r: request: GET /favicon.ico 127.0.0.1 404",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 33",
        "slot update_slots: id  0 | task 0 | kv cache rm [0, end)",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 33, n_tokens = 33, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 33, n_tokens = 33",
        "slot      release: id  0 | task 0 | stop processing: n_past = 742, truncated = 0",
        "slot print_timing: id  0 | task 0 |",
        "prompt eval time =     990.30 ms /    33 tokens",
        "eval time =   19521.92 ms /   710 tokens",
        "total time =   20512.22 ms /   743 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200"
    ]

    parser = LlamaLogParser()
    status_info = parser.parse_multiple_lines(timing_logs)
    
    assert status_info.status == ModelStatus.COMPLETED, f"Expected COMPLETED status, got {status_info.status}"
    assert status_info.prompt_tokens == 33, f"Expected 33 prompt tokens, got {status_info.prompt_tokens}"
    assert status_info.generated_tokens == 710, f"Expected 710 generated tokens, got {status_info.generated_tokens}"
    assert status_info.total_tokens == 743, f"Expected 743 total tokens, got {status_info.total_tokens}"
    assert abs(status_info.processing_speed - 33.32) < 1, f"Expected ~33.32 tokens/s processing speed, got {status_info.processing_speed}" # type: ignore
    assert abs(status_info.generation_speed - 36.37) < 1, f"Expected ~36.37 tokens/s generation speed, got {status_info.generation_speed}" # type: ignore

def test_status_transitions_between_generations():
    """Test that status transitions correctly between multiple generations."""
    full_log = [
        "main: server is listening on http://127.0.0.1:42753 - starting the main loop",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: GET / 127.0.0.1 200",
        "srv  log_server_r: request: GET /props 127.0.0.1 200",
        "srv  log_server_r: request: GET /favicon.ico 127.0.0.1 404",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 33",
        "slot update_slots: id  0 | task 0 | kv cache rm [0, end)",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 33, n_tokens = 33, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 33, n_tokens = 33",
        "slot      release: id  0 | task 0 | stop processing: n_past = 2310, truncated = 0",
        "slot print_timing: id  0 | task 0 |",
        "prompt eval time =    1005.19 ms /    33 tokens",
        "eval time =   61539.85 ms /  2278 tokens",
        "total time =   62545.04 ms /  2311 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 2279 | processing task",
        "slot update_slots: id  0 | task 2279 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 2337",
        "slot update_slots: id  0 | task 2279 | kv cache rm [2310, end)",
        "slot update_slots: id  0 | task 2279 | prompt processing progress, n_past = 2337, n_tokens = 27, progress = 0.011553",
        "slot update_slots: id  0 | task 2279 | prompt done, n_past = 2337, n_tokens = 27",
        "slot      release: id  0 | task 2279 | stop processing: n_past = 5986, truncated = 0",
        "slot print_timing: id  0 | task 2279 |",
        "prompt eval time =     322.48 ms /    27 tokens",
        "eval time =  106688.57 ms /  3650 tokens",
        "total time =  107011.05 ms /  3677 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200"
    ]

    parser = LlamaLogParser()
    status = None
    status_history = []

    for line in full_log:
        status = parser.parse_log_line(line, status)
        status_history.append(status.status)

    # Verify status transitions
    # First generation should complete
    assert ModelStatus.COMPLETED in status_history, "First generation should reach COMPLETED status"
    
    # After completion, we should transition to a new state (could be IDLE or STARTING for next task)
    completed_idx = status_history.index(ModelStatus.COMPLETED)
    assert completed_idx + 1 < len(status_history), "Should have a status after COMPLETED"
    
    # Find the start of the second generation
    second_gen_start = full_log.index("srv  params_from_: Chat format: Hermes 2 Pro", 18)
    second_gen_statuses = status_history[second_gen_start:]
    
    assert ModelStatus.STARTING in second_gen_statuses, "Second generation should have STARTING status"
    assert ModelStatus.PROCESSING_PROMPT in second_gen_statuses, "Second generation should have PROCESSING_PROMPT status"
    assert ModelStatus.GENERATING_RESPONSE in second_gen_statuses, "Second generation should have GENERATING_RESPONSE status"
    assert ModelStatus.COMPLETED in second_gen_statuses, "Second generation should have COMPLETED status"
    assert status_history[-1] == ModelStatus.IDLE, "Final status should be IDLE"

if __name__ == "__main__":
    test_log_parser()
    test_timing_log_parser()
    test_status_transitions_between_generations()
    print("All tests passed!")