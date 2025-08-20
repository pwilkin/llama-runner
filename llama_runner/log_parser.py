import re
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class ModelStatus(Enum):
    IDLE = "Idle"
    STARTING = "Starting"
    PROCESSING_PROMPT = "Processing prompt"
    GENERATING_RESPONSE = "Generating response"
    COMPLETED = "Completed"

@dataclass
class ModelStatusInfo:
    status: ModelStatus
    progress: Optional[float] = None
    processing_speed: Optional[float] = None
    generation_speed: Optional[float] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class LlamaLogParser:
    """Parser for llama.cpp server logs to extract model status information."""

    def __init__(self):
        self.patterns = {
            'new_prompt': re.compile(r'new prompt, n_ctx_slot = \d+, n_keep = \d+, n_prompt_tokens = (\d+)'),
            'prompt_progress': re.compile(r'prompt processing progress, n_past = (\d+), n_tokens = (\d+), progress = ([\d.]+)'),
            'prompt_done': re.compile(r'prompt done, n_past = (\d+), n_tokens = (\d+)'),
            'prompt_eval_time': re.compile(r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens'),
            'eval_time': re.compile(r'eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens'),
            'all_idle': re.compile(r'all slots are idle'),
            'processing_task': re.compile(r'processing task'),
        }
        # For tracking timing information across multiple lines
        self.pending_timing_info = {}
        # For debugging
        self.debug = False

    def parse_log_line(self, line: str, current_status: Optional[ModelStatusInfo] = None) -> ModelStatusInfo:
        """Parse a single log line and return updated status information."""
        if current_status is None:
            current_status = ModelStatusInfo(status=ModelStatus.IDLE)

        if self.debug:
            print(f"DEBUG: Parsing line: {line[:50]}...")
            print(f"DEBUG: Current status: {current_status.status.value}")

        # Check for new prompt - this should reset timing info and start processing
        if 'new prompt' in line:
            match = self.patterns['new_prompt'].search(line)
            if match:
                prompt_tokens = int(match.group(1))
                if self.debug:
                    print(f"DEBUG: Found new prompt with {prompt_tokens} tokens")
                # Reset timing info when starting a new prompt
                self.pending_timing_info = {}
                return ModelStatusInfo(
                    status=ModelStatus.STARTING,
                    prompt_tokens=prompt_tokens
                )

        # Check for prompt processing progress
        if 'prompt processing progress' in line:
            match = self.patterns['prompt_progress'].search(line)
            if match:
                n_past = int(match.group(1))
                n_tokens = int(match.group(2))
                progress = float(match.group(3))
                if self.debug:
                    print(f"DEBUG: Prompt processing progress: {progress*100:.1f}%")
                return ModelStatusInfo(
                    status=ModelStatus.PROCESSING_PROMPT,
                    progress=progress * 100,
                    prompt_tokens=n_tokens
                )

        # Check for prompt done
        if 'prompt done' in line:
            match = self.patterns['prompt_done'].search(line)
            if match:
                n_past = int(match.group(1))
                n_tokens = int(match.group(2))
                if self.debug:
                    print(f"DEBUG: Prompt done")
                return ModelStatusInfo(
                    status=ModelStatus.GENERATING_RESPONSE,
                    prompt_tokens=n_tokens
                )

        # Check for timing information - collect timing data across multiple lines
        if 'prompt eval time' in line:
            match = self.patterns['prompt_eval_time'].search(line)
            if match:
                self.pending_timing_info['prompt_eval_time'] = float(match.group(1))
                self.pending_timing_info['prompt_tokens'] = int(match.group(2))
                if self.debug:
                    print(f"DEBUG: Found prompt eval time: {float(match.group(1))}ms for {int(match.group(2))} tokens")

        if 'eval time' in line and 'prompt eval time' not in line:
            match = self.patterns['eval_time'].search(line)
            if match:
                self.pending_timing_info['eval_time'] = float(match.group(1))
                self.pending_timing_info['generated_tokens'] = int(match.group(2))
                if self.debug:
                    print(f"DEBUG: Found eval time: {float(match.group(1))}ms for {int(match.group(2))} tokens")

        # If we have all the timing information, compute speeds and mark as completed
        if ('prompt_eval_time' in self.pending_timing_info and
            'eval_time' in self.pending_timing_info and
            'prompt_tokens' in self.pending_timing_info and
            'generated_tokens' in self.pending_timing_info):

            prompt_eval_time = self.pending_timing_info['prompt_eval_time']
            eval_time = self.pending_timing_info['eval_time']
            prompt_tokens = self.pending_timing_info['prompt_tokens']
            generated_tokens = self.pending_timing_info['generated_tokens']

            processing_speed = (prompt_tokens / prompt_eval_time) * 1000 if prompt_eval_time > 0 else 0
            generation_speed = (generated_tokens / eval_time) * 1000 if eval_time > 0 else 0

            if self.debug:
                print(f"DEBUG: Computing completion - proc_speed: {processing_speed:.1f}, gen_speed: {generation_speed:.1f}")

            # Clear pending timing info
            self.pending_timing_info = {}

            return ModelStatusInfo(
                status=ModelStatus.COMPLETED,
                processing_speed=processing_speed,
                generation_speed=generation_speed,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens
            )

        # Check for idle state - this should also reset timing info
        if 'all slots are idle' in line:
            if self.debug:
                print(f"DEBUG: Found idle state")
            # Clear pending timing info when going idle
            self.pending_timing_info = {}
            return ModelStatusInfo(status=ModelStatus.IDLE)

        # Check for processing task (transition from idle to starting) - this should reset timing info
        if 'processing task' in line and current_status.status == ModelStatus.IDLE:
            if self.debug:
                print(f"DEBUG: Found processing task, transitioning from IDLE to STARTING")
            # Reset timing info when starting a new task
            self.pending_timing_info = {}
            return ModelStatusInfo(status=ModelStatus.STARTING)

        # If we're in COMPLETED state and we get a new task or prompt, transition to STARTING
        if current_status.status == ModelStatus.COMPLETED:
            if 'processing task' in line or 'new prompt' in line:
                if self.debug:
                    print(f"DEBUG: Found new task/prompt while COMPLETED, transitioning to STARTING")
                self.pending_timing_info = {}  # Reset timing info
                if 'new prompt' in line:
                    match = self.patterns['new_prompt'].search(line)
                    if match:
                        prompt_tokens = int(match.group(1))
                        return ModelStatusInfo(
                            status=ModelStatus.STARTING,
                            prompt_tokens=prompt_tokens
                        )
                return ModelStatusInfo(status=ModelStatus.STARTING)

        if self.debug:
            print(f"DEBUG: No status change, returning current status")
        return current_status

    def parse_multiple_lines(self, lines: list[str]) -> ModelStatusInfo:
        """Parse multiple log lines and return the most recent status."""
        if self.debug:
            print(f"DEBUG: parse_multiple_lines called with {len(lines)} lines")

        # Look for timing information in the full log by finding prompt eval and eval time lines
        full_log = "\n".join(lines)
        prompt_eval_matches = list(self.patterns['prompt_eval_time'].finditer(full_log))
        eval_matches = list(self.patterns['eval_time'].finditer(full_log))

        # Filter out eval matches that are actually prompt eval matches
        filtered_eval_matches = []
        for match in eval_matches:
            # Check if this match is part of a prompt eval time line
            start, end = match.span()
            # Look for the line containing this match
            line_start = full_log.rfind("\n", 0, start) + 1
            line_end = full_log.find("\n", start)
            if line_end == -1:
                line_end = len(full_log)
            line = full_log[line_start:line_end]
            if not line.startswith('prompt eval time'):
                filtered_eval_matches.append(match)

        # Check if we have timing information
        if prompt_eval_matches and filtered_eval_matches:
            # Get the position of the last timing information
            last_prompt_eval = prompt_eval_matches[-1]
            last_eval = filtered_eval_matches[-1]
            last_timing_pos = max(last_prompt_eval.end(), last_eval.end())

            # Check if there are any task-initiating events AFTER the timing information
            has_newer_tasks = False
            for line in lines:
                if (('processing task' in line or 'new prompt' in line) and
                    full_log.find(line) > last_timing_pos):
                    has_newer_tasks = True
                    if self.debug:
                        print(f"DEBUG: Found newer task after timing info: {line[:50]}")
                    break

            # If no newer tasks found after timing info, use timing-based COMPLETED status
            if not has_newer_tasks:
                prompt_eval_time = float(last_prompt_eval.group(1))
                prompt_tokens = int(last_prompt_eval.group(2))
                eval_time = float(last_eval.group(1))
                generated_tokens = int(last_eval.group(2))

                processing_speed = (prompt_tokens / prompt_eval_time) * 1000 if prompt_eval_time > 0 else 0
                generation_speed = (generated_tokens / eval_time) * 1000 if eval_time > 0 else 0

                result = ModelStatusInfo(
                    status=ModelStatus.COMPLETED,
                    processing_speed=processing_speed,
                    generation_speed=generation_speed,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    total_tokens=prompt_tokens + generated_tokens
                )

                if self.debug:
                    print(f"DEBUG: parse_multiple_lines returning COMPLETED with speeds: {processing_speed:.1f}, {generation_speed:.1f}")
                return result

        # If no timing info or newer tasks exist, process line by line
        status = ModelStatusInfo(status=ModelStatus.IDLE)
        self.pending_timing_info = {}  # Reset timing info
        for line in lines:
            status = self.parse_log_line(line, status)

        if self.debug:
            print(f"DEBUG: parse_multiple_lines returning line-by-line status: {status.status.value}")
        return status

    def format_status_text(self, status_info: ModelStatusInfo) -> str:
        """Format status information for display."""
        if status_info.status == ModelStatus.IDLE:
            return "Idle"
        elif status_info.status == ModelStatus.STARTING:
            return "Starting"
        elif status_info.status == ModelStatus.PROCESSING_PROMPT:
            if status_info.progress is not None:
                return f"Processing prompt: {status_info.progress:.1f}%"
            return "Processing prompt"
        elif status_info.status == ModelStatus.GENERATING_RESPONSE:
            return "Generating response"
        elif status_info.status == ModelStatus.COMPLETED:
            if status_info.processing_speed and status_info.generation_speed:
                return f"Generated, speed {status_info.processing_speed:.1f} t/s (proc), {status_info.generation_speed:.1f} t/s (gen)"
            return "Generated"

        return status_info.status.value