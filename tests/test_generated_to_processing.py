import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_generated_to_processing():
    \"\"\"Test the specific scenario: Generated status should transition to Processing prompt.\"\"\"
    
    parser = LlamaLogParser()
    
    # First, get to a Generated status
    first_generation_lines = [
        'slot launch_slot_: id  0 | task 0 | processing task',
        'slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 33',
        'slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 33, n_tokens = 33, progress = 1.000000',
        'slot update_slots: id  0 | task 0 | prompt done, n_past = 33, n_tokens = 33',
        'prompt eval time =    1005.19 ms /    33 tokens',
        'eval time =   61539.85 ms /  2278 tokens',
    ]
    
    print("=== First Generation ===")
    status = None
    for line in first_generation_lines:
        status = parser.parse_log_line(line, status)
    
    print(f"Final status after first generation: {status.status.value}")
    display_text = parser.format_status_text(status)
    print(f"Display text: {display_text}")
    
    # Now simulate what happens in the real app - we get a Generated status
    # and then we want to transition to Processing prompt for the next task
    if status.status == ModelStatus.COMPLETED:
        print("\n=== Simulating Real App Scenario ===")
        print("We have a Generated status. Now let's see what happens when we get new log lines...")
        
        # These are the log lines that should trigger a transition to Processing prompt
        second_generation_lines = [
            'slot launch_slot_: id  0 | task 1 | processing task',
            'slot update_slots: id  0 | task 1 | new prompt, n_ctx_slot = 65024, n_keep = 0, n_prompt_tokens = 2337',
            'slot update_slots: id  0 | task 1 | prompt processing progress, n_past = 2337, n_tokens = 27, progress = 0.011553',
        ]
        
        # Enable debug to see what's happening
        parser.debug = True
        
        for i, line in enumerate(second_generation_lines):
            print(f"\n--- Processing line {i}: {line[:50]} ---")
            new_status = parser.parse_log_line(line, status)
            
            if new_status.status != status.status:
                print(f"STATUS CHANGE: {status.status.value} -> {new_status.status.value}")
                new_display_text = parser.format_status_text(new_status)
                print(f"New display text: {new_display_text}")
            else:
                print(f"NO STATUS CHANGE: {new_status.status.value}")
                new_display_text = parser.format_status_text(new_status)
                print(f"Display text: {new_display_text}")
            
            status = new_status

if __name__ == "__main__":
    test_generated_to_processing()