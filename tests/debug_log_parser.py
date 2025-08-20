import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def debug_log_parser():
    # Debug the log parser with a scenario that reproduces the issue.
    
    # Simulate the exact scenario: get Generated status, then try to transition to Processing
    log_lines = [
        # First generation - should end with Generated status
        'slot launch_slot_: id  0 | task 0 | processing task',
        'slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 33',
        'slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 33, n_tokens = 33, progress = 1.000000',
        'slot update_slots: id  0 | task 0 | prompt done, n_past = 33, n_tokens = 33',
        'prompt eval time =    1005.19 ms /    33 tokens',
        'eval time =   61539.85 ms /  2278 tokens',
        'total time =   62545.04 ms /  2311 tokens',
        
        # Now try to start second generation - this is where the issue occurs
        'slot launch_slot_: id  0 | task 1 | processing task',
        'slot update_slots: id  0 | task 1 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 2337',
        'slot update_slots: id  0 | task 1 | prompt processing progress, n_past = 2337, n_tokens = 27, progress = 0.011553',
        'slot update_slots: id  0 | task 1 | prompt done, n_past = 2337, n_tokens = 27',
        'prompt eval time =     322.48 ms /    27 tokens',
        'eval time =  106688.57 ms /  3650 tokens',
        'total time =  107011.05 ms /  3677 tokens',
    ]

    parser = LlamaLogParser()
    parser.debug = True  # Enable debug logging
    
    print("=== Debugging Log Parser ===")
    status = None
    
    for i, line in enumerate(log_lines):
        print(f"\n--- Line {i}: {line[:50]} ---")
        new_status = parser.parse_log_line(line, status)
        
        # Check if status changed
        if new_status.status != (status.status if status else None):
            status_text = parser.format_status_text(new_status)
            print(f"STATUS CHANGE: {status.status.value if status else 'None'} -> {new_status.status.value}")
            print(f"DISPLAY TEXT: {status_text}")
        else:
            status_text = parser.format_status_text(new_status)
            print(f"NO CHANGE: {new_status.status.value}")
            print(f"DISPLAY TEXT: {status_text}")
        
        status = new_status

if __name__ == "__main__":
    debug_log_parser()