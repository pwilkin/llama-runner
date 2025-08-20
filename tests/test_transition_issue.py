import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_transition_issue():
    # Test the specific transition issue.
    
    # Focus on the transition from first completion to second task start
    critical_lines = [
        "prompt eval time =     171.07 ms /    11 tokens (   15.55 ms per token,    64.30 tokens per second)",
        "eval time =     421.41 ms /    17 tokens (   24.79 ms per token,    40.34 tokens per second)",
        "total time =     592.48 ms /    28 tokens",
        "srv  update_slots: all slots are idle",  # This will cause IDLE status
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 18 | processing task",  # This should cause STARTING status
        "slot update_slots: id  0 | task 18 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 51",
    ]

    parser = LlamaLogParser()
    
    print("=== Testing Critical Transition ===")
    
    # Start with a COMPLETED status (simulating the end of first generation)
    status = parser.parse_log_line("eval time =     421.41 ms /    17 tokens", None)
    print(f"Initial status: {status.status.value}")
    print(f"Initial display: {parser.format_status_text(status)}")
    
    for i, line in enumerate(critical_lines):
        print(f"\n--- Processing: {line[:50]} ---")
        new_status = parser.parse_log_line(line, status)
        
        if new_status.status != status.status:
            print(f"STATUS CHANGE: {status.status.value} -> {new_status.status.value}")
            print(f"Display text: {parser.format_status_text(new_status)}")
        else:
            display_text = parser.format_status_text(new_status)
            print(f"No status change: {new_status.status.value}")
            print(f"Display text: {display_text}")
        
        status = new_status

if __name__ == "__main__":
    test_transition_issue()