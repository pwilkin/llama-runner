import os
import sys
import subprocess
import logging
import asyncio
from typing import Optional, Dict

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QListWidget, QStackedWidget)
from PySide6.QtCore import Slot, Qt, QEvent

from llama_runner.config_loader import load_config
from llama_runner.lmstudio_proxy_thread import LMStudioProxyServer
from llama_runner.ollama_proxy_thread import OllamaProxyServer
from llama_runner import gguf_metadata

from llama_runner.model_status_widget import ModelStatusWidget
from llama_runner.llama_runner_manager import LlamaRunnerManager
from PySide6.QtCore import Signal


class MainWindow(QWidget):
    runner_port_ready_for_proxy = Signal(str, int)
    runner_stopped_for_proxy = Signal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Llama Runner")
        self.resize(800, 600)

        self.config = load_config()
        
        # Load settings from config with defaults
        self.prompt_logging_enabled = self.config.get('logging', {}).get('prompt_logging_enabled', False)
        self.llama_runtimes = self.config.get("llama-runtimes", {})
        self.default_runtime = self.config.get("default_runtime", "llama-server")
        self.models = self.config.get("models", {})
        self.concurrent_runners_limit = self.config.get("concurrentRunners", 1)
        if not isinstance(self.concurrent_runners_limit, int) or self.concurrent_runners_limit < 1:
            logging.warning(f"Invalid 'concurrentRunners' value in config: {self.concurrent_runners_limit}. Defaulting to 1.")
            self.concurrent_runners_limit = 1

        # Proxy settings
        proxies_config = self.config.get('proxies', {})
        self.ollama_proxy_enabled = proxies_config.get('ollama', {}).get('enabled', True)
        lmstudio_proxy_config = proxies_config.get('lmstudio', {})
        self.lmstudio_proxy_enabled = lmstudio_proxy_config.get('enabled', True)
        self.lmstudio_api_key = lmstudio_proxy_config.get('api_key', None)

        self.prompts_logger = logging.getLogger("prompts")

        gguf_metadata.ensure_cache_dir_exists()
        self.model_metadata_cache = {}
        for model_name, model_config in self.models.items():
            model_path = model_config.get("model_path")
            if model_path:
                metadata = gguf_metadata.get_model_lmstudio_format(model_name, model_path, model_config, False)
                if metadata:
                    self.model_metadata_cache[model_name] = metadata
            else:
                logging.warning(f"Model '{model_name}' has no 'model_path' in config. Skipping metadata caching.")

        self.lmstudio_proxy_server: Optional[LMStudioProxyServer] = None
        self.ollama_proxy_server: Optional[OllamaProxyServer] = None

        self.main_layout = QVBoxLayout() # Renamed self.layout to self.main_layout
        self.top_layout = QHBoxLayout()

        self.model_list_widget = QListWidget()
        self.model_list_widget.setMinimumWidth(150)
        self.model_list_widget.setStyleSheet("""
            QListWidget {
                border: none;
                outline: none;
                border-radius: 5px;
                padding: 5px;
                background-color: #f0f0f0;
                font-size: 12pt;
            }
            QListWidget::item {
                padding: 8px;
                margin-bottom: 4px;
                background-color: #f0f0f0;
                border: none;
                outline: none;
            }
            QListWidget::item:selected {
                background-color: #a0c0f0;
                color: #333333;
                border: none;
                outline: none;
            }
            QListWidget::item:selected:focus {
                show-decoration-selected: false;
            }
        """)
        self.main_layout.addLayout(self.top_layout) # Use main_layout

        self.model_status_stack = QStackedWidget()
        self.top_layout.addWidget(self.model_list_widget) # model_list_widget is part of top_layout
        self.top_layout.addWidget(self.model_status_stack) # model_status_stack is part of top_layout

        self.model_status_widgets: Dict[str, ModelStatusWidget] = {}

        for model_name in self.models.keys():
            self.model_list_widget.addItem(model_name)
            model_metadata = self.model_metadata_cache.get(model_name)
            status_widget = ModelStatusWidget(model_name, metadata=model_metadata)
            self.model_status_stack.addWidget(status_widget)
            self.model_status_widgets[model_name] = status_widget
            status_widget.start_button.clicked.connect(lambda checked, name=model_name: self.llama_runner_manager.request_runner_start(name))
            status_widget.stop_button.clicked.connect(lambda checked, name=model_name: self.llama_runner_manager.stop_llama_runner(name))

        self.model_list_widget.currentItemChanged.connect(self.on_model_selection_changed)

        self.no_model_selected_widget = QWidget()
        no_model_layout = QVBoxLayout(self.no_model_selected_widget)
        no_model_label = QLabel("Select a model from the list.")
        no_model_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Corrected Qt.AlignCenter
        no_model_layout.addWidget(no_model_label)
        no_model_layout.addStretch()
        self.model_status_stack.addWidget(self.no_model_selected_widget)
        self.model_status_stack.setCurrentWidget(self.no_model_selected_widget)

        # self.main_layout.addLayout(self.top_layout) # top_layout is already added to main_layout
        self.edit_config_button = QPushButton("Edit config")
        self.main_layout.addWidget(self.edit_config_button) # Use main_layout
        self.setLayout(self.main_layout) # Use main_layout
        self.edit_config_button.clicked.connect(self.open_config_file)

        # --- LlamaRunnerManager instantiation ---
        self.llama_runner_manager = LlamaRunnerManager(
            models=self.models,
            llama_runtimes=self.llama_runtimes,
            default_runtime=self.default_runtime,
            on_started=self.on_runner_started,
            on_stopped=self.on_runner_stopped,
            on_error=self.on_runner_error,
            on_port_ready=self.on_runner_port_ready,
        )
        self.llama_runner_manager.set_concurrent_runners_limit(self.concurrent_runners_limit)


        # --- Create Proxy Server instances ---
        if self.lmstudio_proxy_enabled:
            self.lmstudio_proxy_server = LMStudioProxyServer(
                all_models_config=self.models,
                runtimes_config=self.llama_runtimes,
                is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
                on_runner_port_ready=self.on_runner_port_ready,
                on_runner_stopped=self.on_runner_stopped,
            )

        if self.ollama_proxy_enabled:
            self.ollama_proxy_server = OllamaProxyServer(
                all_models_config=self.models,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
            )


        self.setStyleSheet(self.styleSheet() + """
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 10pt;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

    def closeEvent(self, event):
        """
        Handles the window close event. Stops all running threads and waits for their completion.
        Enhanced to gracefully handle KeyboardInterrupt and ensure all threads are stopped and joined.
        """
        print("MainWindow closing. Stopping all runners and proxy...")

        try:
            # The new manager uses an async stop method.
            # In a Qt closeEvent, we can't easily run an async method to completion.
            # The best we can do is to signal the threads to stop.
            # The qasync loop in main.py will handle the async cleanup.
            self.llama_runner_manager.stop_all_llama_runners_async()

            if self.lmstudio_proxy_server:
                self.lmstudio_proxy_server.stop()
            if self.ollama_proxy_server:
                self.ollama_proxy_server.stop()

        except KeyboardInterrupt:
            print("KeyboardInterrupt received during shutdown. Attempting best-effort thread cleanup...")
            import traceback
            traceback.print_exc()
            # Attempt to stop/join threads again, ignoring further interrupts
            try:
                self.llama_runner_manager.stop_all_llama_runners()
            except Exception as e:
                print(f"Exception during forced runner shutdown: {e}")
            try:
                if self.fastapi_proxy_thread and self.fastapi_proxy_thread.isRunning():
                    self.fastapi_proxy_thread.stop()
                    self.fastapi_proxy_thread.wait()
            except Exception as e:
                print(f"Exception during forced FastAPI proxy shutdown: {e}")
            try:
                if self.ollama_proxy_thread and self.ollama_proxy_thread.isRunning():
                    self.ollama_proxy_thread.stop()
                    self.ollama_proxy_thread.wait()
            except Exception as e:
                print(f"Exception during forced Ollama proxy shutdown: {e}")
        except Exception as e:
            print(f"Exception during shutdown: {e}")
            import traceback
            traceback.print_exc()
        finally:
            event.accept()

    @Slot(str)
    def on_model_selection_changed(self, current_item, previous_item):
        """
        Slot to handle selection changes in the model list.
        Switches the stacked widget to show the selected model's status.
        """
        if current_item:
            model_name = current_item.text()
            if model_name in self.model_status_widgets:
                self.model_status_stack.setCurrentWidget(self.model_status_widgets[model_name])
            else:
                logging.error(f"Status widget not found for selected model: {model_name}")
                self.model_status_stack.setCurrentWidget(self.no_model_selected_widget)
        else:
            self.model_status_stack.setCurrentWidget(self.no_model_selected_widget)

    # Runner management methods moved to LlamaRunnerManager

    # --- Callback methods for LlamaRunnerManager ---

    def on_runner_started(self, model_name: str):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Starting...")
            widget.set_buttons_enabled(False, False)

    def on_runner_stopped(self, model_name: str):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Not Running")
            widget.update_port("N/A")
            widget.set_buttons_enabled(True, False)
        # The new proxy servers don't need to be notified this way.
        # Their internal logic handles runner state.

    def on_runner_error(self, model_name: str, message: str, output_buffer: list):
        from llama_runner.error_output_dialog import ErrorOutputDialog
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Error")

        dialog_message = f"Llama.cpp server for {model_name} encountered an error:\n{message}"
        error_dialog = ErrorOutputDialog(
            title=f"Llama Runner Error: {model_name}",
            message=dialog_message,
            output_lines=output_buffer,
            parent=self
        )
        error_dialog.exec()

    def on_runner_port_ready(self, model_name: str, port: int):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_port(port)
            widget.update_status("Running")
            widget.set_buttons_enabled(False, True)
        # The new proxy servers don't need to be notified this way.


    # --- Callback methods for LlamaRunnerManager ---

    def on_runner_started(self, model_name: str):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Starting...")
            widget.set_buttons_enabled(False, False)

    def on_runner_stopped(self, model_name: str):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Not Running")
            widget.update_port("N/A")
            widget.set_buttons_enabled(True, False)
        self.runner_stopped_for_proxy.emit(model_name)

    def on_runner_error(self, model_name: str, message: str, output_buffer: list):
        from llama_runner.error_output_dialog import ErrorOutputDialog
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Error")

        dialog_message = f"Llama.cpp server for {model_name} encountered an error:\n{message}"
        error_dialog = ErrorOutputDialog(
            title=f"Llama Runner Error: {model_name}",
            message=dialog_message,
            output_lines=output_buffer,
            parent=self
        )
        error_dialog.exec()

    def on_runner_port_ready(self, model_name: str, port: int):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_port(port)
            widget.update_status("Running")
            widget.set_buttons_enabled(False, True)
        self.runner_port_ready_for_proxy.emit(model_name, port)

    def open_config_file(self):
        """
        Opens the config.json file in the system's default editor.
        """
        config_path = os.path.expanduser("~/.llama-runner/config.json")
        logging.info(f"Attempting to open config file: {config_path}")

        if not os.path.exists(config_path):
            logging.error(f"Config file not found: {config_path}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(config_path)
            elif sys.platform == "darwin": # macOS
                subprocess.run(["open", config_path], check=True)
            else: # linux variants
                subprocess.run(["xdg-open", config_path], check=True)
            logging.info(f"Successfully opened config file: {config_path}")
        except FileNotFoundError:
            logging.error(f"Could not find command to open file on {sys.platform}.")
        except Exception as e:
            logging.error(f"Error opening config file {config_path}: {e}")
