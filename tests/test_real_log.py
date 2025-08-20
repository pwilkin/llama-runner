import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_real_log():
    # Test with the actual log data provided.
    
    real_log = [
        "main: server is listening on http://127.0.0.1:42443 - starting the main loop",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: GET / 127.0.0.1 200",
        "srv  log_server_r: request: GET /props 127.0.0.1 200",
        "srv  log_server_r: request: GET /favicon.ico 127.0.0.1 404",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 11",
        "slot update_slots: id  0 | task 0 | kv cache rm [0, end)",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 11, n_tokens = 11, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 11, n_tokens = 11",
        "slot      release: id  0 | task 0 | stop processing: n_past = 27, truncated = 0",
        "slot print_timing: id  0 | task 0 |",
        "prompt eval time =     171.07 ms /    11 tokens (   15.55 ms per token,    64.30 tokens per second)",
        "eval time =     421.41 ms /    17 tokens (   24.79 ms per token,    40.34 tokens per second)",
        "total time =     592.48 ms /    28 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 18 | processing task",
        "slot update_slots: id  0 | task 18 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 51",
        "slot update_slots: id  0 | task 18 | kv cache rm [27, end)",
        "slot update_slots: id  0 | task 18 | prompt processing progress, n_past = 51, n_tokens = 24, progress = 0.470588",
        "slot update_slots: id  0 | task 18 | prompt done, n_past = 51, n_tokens = 24",
        "slot      release: id  0 | task 18 | stop processing: n_past = 316, truncated = 0",
        "slot print_timing: id  0 | task 18 |",
        "prompt eval time =     291.62 ms /    24 tokens (   12.15 ms per token,    82.30 tokens per second)",
        "eval time =    7190.50 ms /   266 tokens (   27.03 ms per token,    36.99 tokens per second)",
        "total time =    7482.12 ms /   290 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 285 | processing task",
        "slot update_slots: id  0 | task 285 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 331",
        "slot update_slots: id  0 | task 285 | kv cache rm [316, end)",
        "slot update_slots: id  0 | task 285 | prompt processing progress, n_past = 331, n_tokens = 15, progress = 0.045317",
        "slot update_slots: id  0 | task 285 | prompt done, n_past = 331, n_tokens = 15",
        "slot      release: id  0 | task 285 | stop processing: n_past = 618, truncated = 0",
        "slot print_timing: id  0 | task 285 |",
        "prompt eval time =     230.74 ms /    15 tokens (   15.38 ms per token,    65.01 tokens per second)",
        "eval time =    8103.90 ms /   288 tokens (   28.14 ms per token,    35.54 tokens per second)",
        "total time =    8334.64 ms /   303 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 574 | processing task",
        "slot update_slots: id  0 | task 574 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 634",
        "slot update_slots: id  0 | task 574 | kv cache rm [618, end)",
        "slot update_slots: id  0 | task 574 | prompt processing progress, n_past = 634, n_tokens = 16, progress = 0.025237",
        "slot update_slots: id  0 | task 574 | prompt done, n_past = 634, n_tokens = 16",
        "slot      release: id  0 | task 574 | stop processing: n_past = 981, truncated = 0",
        "slot print_timing: id  0 | task 574 |",
        "prompt eval time =     219.81 ms /    16 tokens (   13.74 ms per token,    72.79 tokens per second)",
        "eval time =   10037.44 ms /   348 tokens (   28.84 ms per token,    34.67 tokens per second)",
        "total time =   10257.25 ms /   364 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 923 | processing task",
        "slot update_slots: id  0 | task 923 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 1005",
        "slot update_slots: id  0 | task 923 | kv cache rm [981, end)",
        "slot update_slots: id  0 | task 923 | prompt processing progress, n_past = 1005, n_tokens = 24, progress = 0.023881",
        "slot update_slots: id  0 | task 923 | prompt done, n_past = 1005, n_tokens = 24",
        "slot      release: id  0 | task 923 | stop processing: n_past = 1453, truncated = 0",
        "slot print_timing: id  0 | task 923 |",
        "prompt eval time =     278.21 ms /    24 tokens (   11.59 ms per token,    86.27 tokens per second)",
        "eval time =   13277.59 ms /   449 tokens (   29.57 ms per token,    33.82 tokens per second)",
        "total time =   13555.80 ms /   473 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 1373 | processing task",
        "slot update_slots: id  0 | task 1373 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 1480",
        "slot update_slots: id  0 | task 1373 | kv cache rm [1453, end)",
        "slot update_slots: id  0 | task 1373 | prompt processing progress, n_past = 1480, n_tokens = 27, progress = 0.018243",
        "slot update_slots: id  0 | task 1373 | prompt done, n_past = 1480, n_tokens = 27",
        "slot      release: id  0 | task 1373 | stop processing: n_past = 1860, truncated = 0",
        "slot print_timing: id  0 | task 1373 |",
        "prompt eval time =     310.00 ms /    27 tokens (   11.48 ms per token,    87.10 tokens per second)",
        "eval time =   11298.09 ms /   381 tokens (   29.65 ms per token,    33.72 tokens per second)",
        "total time =   11608.09 ms /   408 tokens",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 1755 | processing task",
        "slot update_slots: id  0 | task 1755 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 1880",
        "slot update_slots: id  0 | task 1755 | kv cache rm [1860, end)",
        "slot update_slots: id  0 | task 1755 | prompt processing progress, n_past = 1880, n_tokens = 20, progress = 0.010638",
        "slot update_slots: id  0 | task 1755 | prompt done, n_past = 1880, n_tokens = 20",
        "slot      release: id  0 | task 1755 | stop processing: n_past = 2161, truncated = 0",
        "slot print_timing: id  0 | task 1755 |",
        "prompt eval time =     249.57 ms /    20 tokens (   12.48 ms per token,    80.14 tokens per second)",
        "eval time =    8542.56 ms /   282 tokens (   30.29 ms per token,    33.01 tokens per second)",
        "total time =    8792.12 ms /   302 tokens",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 2038 | processing task",
        "slot update_slots: id  0 | task 2038 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 2181",
        "slot update_slots: id  0 | task 2038 | kv cache rm [2161, end)",
        "slot update_slots: id  0 | task 2038 | prompt processing progress, n_past = 2181, n_tokens = 20, progress = 0.009170",
        "slot update_slots: id  0 | task 2038 | prompt done, n_past = 2181, n_tokens = 20"
    ]

    parser = LlamaLogParser()
    
    print("=== Testing Real Log Data ===")
    status = None
    
    for i, line in enumerate(real_log):
        # Print key lines
        if "processing task" in line or "prompt processing progress" in line or "prompt done" in line or "eval time" in line:
            print(f"\n--- Line {i}: {line[:60]} ---")
        
        new_status = parser.parse_log_line(line, status)
        
        # Check if status changed
        if new_status.status != status.status if status else True:
            if "processing task" in line or "prompt processing progress" in line or "prompt done" in line or "eval time" in line:
                status_text = parser.format_status_text(new_status)
                print(f"STATUS CHANGE: {status.status.value if status else 'None'} -> {new_status.status.value}")
                print(f"Display text: {status_text}")
        
        status = new_status
        
        # Check if we're stuck in COMPLETED status
        if status.status == ModelStatus.COMPLETED:
            # Look ahead to see if there's a new task
            if i + 1 < len(real_log):
                next_line = real_log[i + 1]
                if "processing task" in next_line:
                    print(f"*** Found new processing task at line {i+1} while in COMPLETED status ***")

if __name__ == "__main__":
    test_real_log()