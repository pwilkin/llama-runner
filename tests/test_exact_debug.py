import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_exact_debug_scenario():
    # Test the exact scenario from the debug output.
    
    # Create logs that would result in the behavior seen in debug output
    # Multiple completions with timing info but no transition lines
    logs = []
    
    # Add some initial lines
    logs.extend([
        "main: server is listening on http://127.0.0.1:42443 - starting the main loop",
        "srv  update_slots: all slots are idle",
        "srv  log_server_r: request: GET / 127.0.0.1 200",
        "srv  log_server_r: request: GET /props 127.0.0.1 200",
        "srv  log_server_r: request: GET /favicon.ico 127.0.0.1 404",
        "srv  params_from_: Chat format: Hermes 2 Pro",
    ])
    
    # Add multiple generations
    for task_id in [0, 18, 285, 574]:
        logs.extend([
            f"slot launch_slot_: id  0 | task {task_id} | processing task",
            f"slot update_slots: id  0 | task {task_id} | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = {11 + task_id}",
            f"slot update_slots: id  0 | task {task_id} | kv cache rm [0, end)",
            f"slot update_slots: id  0 | task {task_id} | prompt processing progress, n_past = {11 + task_id}, n_tokens = {11 + task_id}, progress = 1.000000",
            f"slot update_slots: id  0 | task {task_id} | prompt done, n_past = {11 + task_id}, n_tokens = {11 + task_id}",
            f"slot      release: id  0 | task {task_id} | stop processing: n_past = {27 + task_id}, truncated = 0",
            f"slot print_timing: id  0 | task {task_id} |",
            f"prompt eval time =     {171.07 + task_id:.2f} ms /    {11 + task_id} tokens",
            f"eval time =     {421.41 + task_id * 100:.2f} ms /    {17 + task_id} tokens",
            f"total time =     {592.48 + task_id * 100:.2f} ms /    {28 + task_id} tokens",
            # Note: No "all slots are idle" lines to simulate the real issue
        ])
    
    # Truncate to 200 lines to simulate what the real application might be doing
    if len(logs) > 200:
        logs = logs[-200:]
    
    parser = LlamaLogParser()
    parser.debug = True
    
    print(f"=== Testing with {len(logs)} log lines ===")
    status = parser.parse_multiple_lines(logs)
    print(f"Final status: {status.status.value}")
    if status.status == ModelStatus.COMPLETED:
        print(f"  Speeds: {status.processing_speed:.1f} t/s (proc), {status.generation_speed:.1f} t/s (gen)")

if __name__ == "__main__":
    test_exact_debug_scenario()