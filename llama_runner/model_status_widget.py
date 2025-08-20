import logging
from typing import Optional, Dict, Any, Callable, List

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QTimer

def human_readable_size(size_in_bytes: Optional[int]) -> str:
    """Formats a size in bytes into a human-readable string (e.g., KB, MB, GB)."""
    if size_in_bytes is None:
        return "N/A"
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / (1024**2):.2f} MB"
    elif size_in_bytes < 1024**4:
        return f"{size_in_bytes / (1024**3):.2f} GB"
    else:
        return f"{size_in_bytes / (1024**4):.2f} TB"

class ModelStatusWidget(QWidget):
    """
    Widget to display status and controls for a single model.
    """
    def __init__(self, model_name: str, metadata: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.metadata = metadata
        self.main_layout = QVBoxLayout()
        
        # Log monitoring
        from llama_runner.log_parser import LlamaLogParser
        self.log_parser = LlamaLogParser()
        self.log_provider_callback: Optional[Callable[[], List[str]]] = None
        self.log_monitor_timer = QTimer()
        self.log_monitor_timer.timeout.connect(self._update_status_from_logs)
        self.log_monitor_timer.setInterval(1000)  # Update every second

        # Metadata section
        self.metadata_layout = QVBoxLayout()
        self.metadata_label = QLabel("<b>Metadata:</b>")
        self.metadata_layout.addWidget(self.metadata_label)

        self.arch_label = QLabel("Architecture: N/A")
        self.metadata_layout.addWidget(self.arch_label)

        self.quant_label = QLabel("Quantization: N/A")
        self.metadata_layout.addWidget(self.quant_label)

        self.size_label = QLabel("Size: N/A")
        self.metadata_layout.addWidget(self.size_label)

        self.main_layout.addLayout(self.metadata_layout) # Add metadata layout to main widget layout

        self.model_label = QLabel(f"<b>{self.model_name}</b>")
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_label.setStyleSheet("font-size: 16pt;") # Larger font for model name
        self.main_layout.addWidget(self.model_label)

        self.status_label = QLabel("Status: Not Running")
        self.main_layout.addWidget(self.status_label)

        self.port_label = QLabel("Port: N/A")
        self.main_layout.addWidget(self.port_label)

        # Real-time status display (above the buttons)
        self.real_time_status_label = QLabel("Real-time: Idle")
        self.real_time_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.real_time_status_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 11pt;
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 8px;
                border-radius: 4px;
                margin: 5px 0px;
            }
        """)
        self.main_layout.addWidget(self.real_time_status_label)

        self.start_button = QPushButton(f"Start {self.model_name}")
        self.main_layout.addWidget(self.start_button)

        self.stop_button = QPushButton(f"Stop {self.model_name}")
        self.stop_button.setEnabled(False) # Initially disabled
        self.main_layout.addWidget(self.stop_button)

        self.main_layout.addStretch() # Push everything to the top
        self.setLayout(self.main_layout)
        # Apply styling to the ModelStatusWidget and its children
        self.setStyleSheet("""
            ModelStatusWidget {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 8px;
                padding: 15px; /* Add padding to the widget */
                margin: 10px; /* Add margin around the widget */
            }
            QLabel {
                font-size: 10pt; /* Default label font size */
                margin-bottom: 5px;
            }
            QLabel:first-child { /* Style for the "Metadata:" label */
                 font-weight: bold;
                 margin-bottom: 10px;
            }
            QLabel[text^="Architecture:"],
            QLabel[text^="Quantization:"],
            QLabel[text^="Size:"] {
                 font-size: 9pt; /* Smaller font for metadata details */
                 color: #999999; /* Lighter gray for metadata text */
                 margin-left: 10px; /* Indent metadata details */
            }
            QLabel[text^="Status:"] {
                 font-weight: bold;
                 margin-top: 10px;
            }
            QLabel[text^="Port:"] {
                 font-weight: bold;
            }
            QPushButton:disabled {
                 border: 1px solid #CCCCCC; /* Lighter border for disabled buttons */
            }
        """)

        # Update metadata display if metadata is provided
        if self.metadata:
            self.update_metadata(self.metadata)

    def update_metadata(self, metadata: Dict[str, Any]):
        """Updates the displayed metadata."""
        self.metadata = metadata
        self.arch_label.setText(f"Architecture: {metadata.get('arch', 'N/A')}")
        self.quant_label.setText(f"Quantization: {metadata.get('quantization', 'N/A')}")
        # Format size nicely, assuming size is in bytes (LM Studio format)
        size_bytes = metadata.get('size', None)
        if size_bytes is not None:
            # Attempt to convert to integer as per user suggestion
            try:
                size_bytes_int = int(size_bytes)
                self.size_label.setText(f"Size: {human_readable_size(size_bytes_int)}")
            except (ValueError, TypeError) as e:
                # Log detailed information about the conversion failure
                actual_type = type(size_bytes).__name__
                logging.warning(f"Size conversion failed for value '{size_bytes}' (type: {actual_type}): {str(e)}")
                # Try to clean the value by removing non-digit characters except decimal point
                if isinstance(size_bytes, str):
                    cleaned = ''.join(filter(lambda x: x.isdigit() or x == '.', size_bytes))
                    if cleaned and cleaned != '.':
                        try:
                            size_bytes_float = float(cleaned)
                            # Convert to bytes if the value has a unit (assuming it's in GB)
                            if 'gb' in size_bytes.lower():
                                size_bytes_int = int(size_bytes_float * (1024 ** 3))
                            elif 'mb' in size_bytes.lower():
                                size_bytes_int = int(size_bytes_float * (1024 ** 2))
                            elif 'kb' in size_bytes.lower():
                                size_bytes_int = int(size_bytes_float * 1024)
                            else:
                                size_bytes_int = int(size_bytes_float)
                            
                            self.size_label.setText(f"Size: {human_readable_size(size_bytes_int)}")
                            return
                        except (ValueError, TypeError):
                            pass  # Fall through to raw value display
                
                # Final fallback: display raw value
                self.size_label.setText(f"Size: {size_bytes}")
        else:
            self.size_label.setText("Size: N/A")

    def update_status(self, status: str):
        self.status_label.setText(f"Status: {status}")

    def update_port(self, port: int | str):
        self.port_label.setText(f"Port: {port}")

    def set_buttons_enabled(self, start_enabled: bool, stop_enabled: bool):
        self.start_button.setEnabled(start_enabled)
        self.stop_button.setEnabled(stop_enabled)

    def set_log_provider(self, log_provider_callback: Callable[[], List[str]]):
        """Set the callback function to get logs from the runner."""
        self.log_provider_callback = log_provider_callback

    def start_log_monitoring(self):
        """Start monitoring logs for status updates."""
        if self.log_provider_callback:
            self.log_monitor_timer.start()

    def stop_log_monitoring(self):
        """Stop monitoring logs."""
        self.log_monitor_timer.stop()

    def _update_status_from_logs(self):
        """Update status based on the latest logs."""
        if not self.log_provider_callback:
            return

        logs = self.log_provider_callback()
        if not logs:
            return  # Keep current status when no new logs

        from llama_runner.log_parser import LlamaLogParser, ModelStatus
        parser = LlamaLogParser()
        status_info = parser.parse_multiple_lines(logs)

        # Format status text for display
        status_text = parser.format_status_text(status_info)

        # Update main status - always update to show current state
        self.update_status(status_text)

        # Update real-time status - for IDLE, show a more user-friendly message
        if status_info.status == ModelStatus.IDLE:
            self.real_time_status_label.setText("Real-time: Waiting for requests...")
        else:
            self.real_time_status_label.setText(f"Real-time: {status_text}")

        # Start/stop monitoring logic
        if any(k in status_text for k in ("Starting", "Processing", "Generating")):
            if not self.log_monitor_timer.isActive():
                self.start_log_monitoring()
        elif any(k in status_text for k in ("Not Running", "Error")):
            if self.log_monitor_timer.isActive():
                self.stop_log_monitoring()