import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_real_application_scenario():
    # Test a scenario that simulates what the real application is doing.
    
    # Simulate the logs that would be available at different times
    # First, we have a completion
    logs_after_first_completion = [
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 11",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 11, n_tokens = 11, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 11, n_tokens = 11",
        "prompt eval time =     171.07 ms /    11 tokens",
        "eval time =     421.41 ms /    17 tokens",
        "total time =     592.48 ms /    28 tokens",
        # These lines might not be in the first batch
    ]
    
    # Then we have the transition to idle and then to starting new task
    logs_with_transition = [
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 11",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 11, n_tokens = 11, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 11, n_tokens = 11",
        "prompt eval time =     171.07 ms /    11 tokens",
        "eval time =     421.41 ms /    17 tokens",
        "total time =     592.48 ms /    28 tokens",
        "srv  update_slots: all slots are idle",  # This should cause transition to IDLE
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 18 | processing task",  # This should cause transition to STARTING
    ]
    
    parser = LlamaLogParser()
    parser.debug = True
    
    print("=== Testing logs after first completion ===")
    status1 = parser.parse_multiple_lines(logs_after_first_completion)
    print(f"Status after first completion: {status1.status.value}")
    if status1.status == ModelStatus.COMPLETED:
        print(f"  Speeds: {status1.processing_speed:.1f} t/s (proc), {status1.generation_speed:.1f} t/s (gen)")
    
    print("\n=== Testing logs with transition ===")
    status2 = parser.parse_multiple_lines(logs_with_transition)
    print(f"Status with transition: {status2.status.value}")
    if status2.status == ModelStatus.COMPLETED:
        print(f"  Speeds: {status2.processing_speed:.1f} t/s (proc), {status2.generation_speed:.1f} t/s (gen)")

if __name__ == "__main__":
    test_real_application_scenario()