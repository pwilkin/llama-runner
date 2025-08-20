#!/usr/bin/env python3
"""
Test for LlamaLogParser using realistic llama.cpp server logs.
"""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama_runner.log_parser import LlamaLogParser, ModelStatus

def test_log_parser_strict():
    parser = LlamaLogParser()
    sample_logs = [
        "srv  log_server_r: request: GET / 127.0.0.1 200",
        "srv  log_server_r: request: GET /props 127.0.0.1 200",
        "srv  log_server_r: request: GET /favicon.ico 127.0.0.1 404",
        "srv  params_from_: Chat format: GPT-OSS",
        "slot launch_slot_: id  0 | task 0 | processing task",
        "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 55040, n_keep = 0, n_prompt_tokens = 70",
        "slot update_slots: id  0 | task 0 | kv cache rm [0, end)",
        "slot update_slots: id  0 | task 0 | prompt processing progress, n_past = 70, n_tokens = 70, progress = 1.000000",
        "slot update_slots: id  0 | task 0 | prompt done, n_past = 70, n_tokens = 70",
        "slot update_slots: id  0 | task 0 | SWA checkpoint create, pos_min = 0, pos_max = 69, size = 1.642 MiB, total = 1/3 (1.642 MiB)",
        "slot      release: id  0 | task 0 | stop processing: n_past = 117, truncated = 0",
        "slot print_timing: id  0 | task 0 |",
        "prompt eval time =    1249.71 ms /    70 tokens (   17.85 ms per token,    56.01 tokens per second)",
        "eval time =    2042.52 ms /    48 tokens (   42.55 ms per token,    23.50 tokens per second)",
        "total time =    3292.23 ms /   118 tokens",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
        "srv  params_from_: Chat format: GPT-OSS",
        "slot launch_slot_: id  0 | task 49 | processing task",
        "slot update_slots: id  0 | task 49 | new prompt, n_ctx_slot = 55040, n_keep = 0, n_prompt_tokens = 114",
        "slot update_slots: id  0 | task 49 | kv cache rm [71, end)",
        "slot update_slots: id  0 | task 49 | prompt processing progress, n_past = 114, n_tokens = 43, progress = 0.377193",
        "slot update_slots: id  0 | task 49 | prompt done, n_past = 114, n_tokens = 43",
        "slot update_slots: id  0 | task 49 | SWA checkpoint create, pos_min = 0, pos_max = 113, size = 2.673 MiB, total = 2/3 (4.315 MiB)",
        "slot      release: id  0 | task 49 | stop processing: n_past = 2084, truncated = 0",
        "slot print_timing: id  0 | task 49 |",
        "prompt eval time =    1054.96 ms /    43 tokens (   24.53 ms per token,    40.76 tokens per second)",
        "eval time =   84423.88 ms /  1971 tokens (   42.83 ms per token,    23.35 tokens per second)",
        "total time =   85478.85 ms /  2014 tokens",
        "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200",
        "srv  update_slots: all slots are idle",
    ]

    # Locate indices for the first prompt events so assertions match parser behavior
    idx_new = next(i for i, l in enumerate(sample_logs) if "new prompt" in l and "task 0" in l)
    status_after_new = parser.parse_multiple_lines(sample_logs[: idx_new + 1])
    assert status_after_new.status == ModelStatus.STARTING
    assert status_after_new.prompt_tokens == 70

    # After processing progress (100%) for the first prompt
    idx_progress = next(i for i, l in enumerate(sample_logs) if "prompt processing progress" in l and "task 0" in l)
    status_after_progress = parser.parse_multiple_lines(sample_logs[: idx_progress + 1])
    assert status_after_progress.status == ModelStatus.PROCESSING_PROMPT
    assert status_after_progress.progress is not None and math.isclose(status_after_progress.progress, 100.0, rel_tol=1e-3)

    # After prompt done -> generating for the first prompt
    idx_done = next(i for i, l in enumerate(sample_logs) if "prompt done" in l and "task 0" in l)
    status_after_done = parser.parse_multiple_lines(sample_logs[: idx_done + 1])
    assert status_after_done.status == ModelStatus.GENERATING_RESPONSE
    assert status_after_done.prompt_tokens == 70

    # Second prompt: check partial progress parsing
    idx_progress2 = sample_logs.index("slot update_slots: id  0 | task 49 | prompt processing progress, n_past = 114, n_tokens = 43, progress = 0.377193")
    status_second_progress = parser.parse_multiple_lines(sample_logs[:idx_progress2+1])
    assert status_second_progress.status == ModelStatus.PROCESSING_PROMPT
    assert status_second_progress.prompt_tokens == 43
    assert status_second_progress.progress is not None and math.isclose(status_second_progress.progress, 0.377193 * 100, rel_tol=1e-6)

    # After second prompt done
    idx_done2 = sample_logs.index("slot update_slots: id  0 | task 49 | prompt done, n_past = 114, n_tokens = 43")
    status_second_done = parser.parse_multiple_lines(sample_logs[:idx_done2+1])
    assert status_second_done.status == ModelStatus.GENERATING_RESPONSE
    assert status_second_done.prompt_tokens == 43

    # Final overall parse should end in idle as logs show all slots are idle at the end
    final_status = parser.parse_multiple_lines(sample_logs)
    assert final_status.status == ModelStatus.IDLE