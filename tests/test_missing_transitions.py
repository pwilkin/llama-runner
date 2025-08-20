import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_missing_transition_lines():
    # Test what happens when transition lines are missing from the logs.
    
    # Logs that contain timing info but no transition to idle or starting
    logs_without_transitions = [
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 11",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 11, n_tokens = 11, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 11, n_tokens = 11",
        "prompt eval time =     171.07 ms /    11 tokens",
        "eval time =     421.41 ms /    17 tokens",
        "total time =     592.48 ms /    28 tokens",
        # Note: No "all slots are idle" or "processing task" lines
    ]
    
    parser = LlamaLogParser()
    parser.debug = True
    
    print("=== Testing logs without transition lines ===")
    status = parser.parse_multiple_lines(logs_without_transitions)
    print(f"Final status: {status.status.value}")
    if status.status == ModelStatus.COMPLETED:
        print(f"  Speeds: {status.processing_speed:.1f} t/s (proc), {status.generation_speed:.1f} t/s (gen)")

if __name__ == "__main__":
    test_missing_transition_lines()