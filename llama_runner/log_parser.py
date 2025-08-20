import re
from typing import Optional, NamedTuple, Dict, Any
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
            'timing': re.compile(r'prompt eval time =\s+([\d.]+) ms / (\d+) tokens.*eval time =\s+([\d.]+) ms / (\d+) tokens'),
            'all_idle': re.compile(r'all slots are idle'),
            'processing_task': re.compile(r'processing task'),
        }
    
    def parse_log_line(self, line: str, current_status: Optional[ModelStatusInfo] = None) -> ModelStatusInfo:
        """Parse a single log line and return updated status information."""
        if current_status is None:
            current_status = ModelStatusInfo(status=ModelStatus.IDLE)
        
        # Check for new prompt
        if 'new prompt' in line:
            match = self.patterns['new_prompt'].search(line)
            if match:
                prompt_tokens = int(match.group(1))
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
                return ModelStatusInfo(
                    status=ModelStatus.GENERATING_RESPONSE,
                    prompt_tokens=n_tokens
                )
        
        # Check for timing information
        if 'prompt eval time' in line and 'eval time' in line:
            match = self.patterns['timing'].search(line)
            if match:
                prompt_eval_time = float(match.group(1))
                prompt_tokens = int(match.group(2))
                eval_time = float(match.group(3))
                generated_tokens = int(match.group(4))
                
                processing_speed = (prompt_tokens / prompt_eval_time) * 1000 if prompt_eval_time > 0 else 0
                generation_speed = (generated_tokens / eval_time) * 1000 if eval_time > 0 else 0
                
                return ModelStatusInfo(
                    status=ModelStatus.COMPLETED,
                    processing_speed=processing_speed,
                    generation_speed=generation_speed,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    total_tokens=prompt_tokens + generated_tokens
                )
        
        # Check for idle state
        if 'all slots are idle' in line:
            return ModelStatusInfo(status=ModelStatus.IDLE)
        
        # Check for processing task (transition from idle to starting)
        if 'processing task' in line and current_status.status == ModelStatus.IDLE:
            return ModelStatusInfo(status=ModelStatus.STARTING)
        
        return current_status
    
    def parse_multiple_lines(self, lines: list[str]) -> ModelStatusInfo:
        """Parse multiple log lines and return the most recent status."""
        status = ModelStatusInfo(status=ModelStatus.IDLE)
        
        for line in lines:
            status = self.parse_log_line(line, status)
        
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