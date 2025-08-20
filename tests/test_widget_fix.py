import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama_runner.log_parser import LlamaLogParser, ModelStatus, ModelStatusInfo

def test_widget_behavior():
    # Test how the widget should behave with status updates.
    
    # Simulate the widget's behavior with our fix
    parser = LlamaLogParser()
    
    # First, get to a Generated status
    first_generation_lines = [
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 64000, n_keep = 0, n_prompt_tokens = 11",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 11, n_tokens = 11, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 11, n_tokens = 11",
        "prompt eval time =     171.07 ms /    11 tokens",
        "eval time =     421.41 ms /    17 tokens",
    ]
    
    print("=== First Generation ===")
    status = None
    for line in first_generation_lines:
        status = parser.parse_log_line(line, status)
    
    print(f"Final status after first generation: {status.status.value}")
    display_text = parser.format_status_text(status)
    print(f"Display text: {display_text}")
    
    # Simulate the widget's OLD behavior (before our fix)
    print("\n=== OLD Widget Behavior (before fix) ===")
    if status.status != ModelStatus.IDLE:
        print(f"Widget would update to: {display_text}")
    else:
        print("Widget would NOT update (keeps previous display)")
    
    # Simulate the transition through IDLE to the next task
    print("\n=== Transition Through IDLE ===")
    transition_lines = [
        "total time =     592.48 ms /    28 tokens",
        "srv  update_slots: all slots are idle",  # This causes IDLE status
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  params_from_: Chat format: Hermes 2 Pro",
        "slot launch_slot_: id  0 | task 18 | processing task",  # This causes STARTING status
    ]
    
    for line in transition_lines:
        status = parser.parse_log_line(line, status)
        display_text = parser.format_status_text(status)
        print(f"Status: {status.status.value:20} | Display: {display_text}")
        
        # Simulate the widget's OLD behavior
        if status.status != ModelStatus.IDLE:
            print(f"  OLD widget: Would update to '{display_text}'")
        else:
            print(f"  OLD widget: Would NOT update (keeps previous display)")
        
        # Simulate the widget's NEW behavior (after our fix)
        print(f"  NEW widget: Always updates to '{display_text}'")

if __name__ == "__main__":
    test_widget_behavior()