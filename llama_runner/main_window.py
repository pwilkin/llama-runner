import os
import sys
import subprocess
import logging
from typing import Optional, Dict

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QListWidget, QStackedWidget)
from PySide6.QtCore import Slot, Qt, QTimer, QEvent

from llama_runner.config_loader import load_config
from llama_runner.lmstudio_proxy_thread import FastAPIProxyThread
from llama_runner.ollama_proxy_thread import OllamaProxyThread
from llama_runner import gguf_metadata

from llama_runner.model_status_widget import ModelStatusWidget
from llama_runner.llama_runner_manager import LlamaRunnerManager # Import events if MainWindow were to handle them directly


class MainWindow(QWidget):
    # runner_port_ready_for_proxy and runner_stopped_for_proxy are now defined in LlamaRunnerManager
    # # Signal emitted when a runner's port is ready
    # # This signal is emitted by MainWindow, connected to the proxy thread
    # runner_port_ready_for_proxy = Signal(str, int)
    # # Signal emitted when a runner stops
    # # This signal is emitted by MainWindow, connected to the proxy thread
    # runner_stopped_for_proxy = Signal(str)

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

        self.fastapi_proxy_thread: Optional[FastAPIProxyThread] = None
        self.ollama_proxy_thread: Optional[OllamaProxyThread] = None

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
            model_status_widgets=self.model_status_widgets,
            # runner_port_ready_for_proxy and runner_stopped_for_proxy are now owned by LlamaRunnerManager
            parent=self,
        )
        self.llama_runner_manager.set_concurrent_runners_limit(self.concurrent_runners_limit)

        # --- Start the FastAPI Proxy (for LM Studio) automatically if enabled ---
        if self.lmstudio_proxy_enabled:
            self.start_fastapi_proxy()
        else:
            print("LM Studio compatible proxy (FastAPI) is disabled in config.")
            self.fastapi_proxy_thread = None # Ensure it's None if not started

        # --- Start the Ollama Proxy automatically if enabled ---
        if self.ollama_proxy_enabled:
            self.start_ollama_proxy()
        else:
            print("Ollama proxy is disabled in config.")
            self.ollama_proxy_thread = None # Ensure it's None if not started

        self._status_check_timer = QTimer(self)
        self._status_check_timer.start(5000)
        self._status_check_timer.timeout.connect(self.llama_runner_manager.check_runner_statuses)

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
            # Stop and wait for all runner threads
            self.llama_runner_manager.stop_all_llama_runners()

            # Stop and wait for FastAPI proxy thread
            if self.fastapi_proxy_thread and self.fastapi_proxy_thread.isRunning():
                self.fastapi_proxy_thread.stop()
                self.fastapi_proxy_thread.wait()

            # Stop and wait for Ollama proxy thread
            if self.ollama_proxy_thread and self.ollama_proxy_thread.isRunning():
                self.ollama_proxy_thread.stop()
                self.ollama_proxy_thread.wait()

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

    def start_fastapi_proxy(self):
        """
        Starts the FastAPI proxy in a separate thread.
        """
        if self.fastapi_proxy_thread is not None and self.fastapi_proxy_thread.isRunning():
            print("FastAPI Proxy is already running.")
            return

        print("Starting FastAPI Proxy (for LM Studio)...")

        self.fastapi_proxy_thread = FastAPIProxyThread(
            all_models_config=self.models,
            runtimes_config=self.llama_runtimes,
            is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
            get_runner_port_callback=self.llama_runner_manager.get_runner_port,
            request_runner_start_callback=self.llama_runner_manager.request_runner_start,
            prompt_logging_enabled=self.prompt_logging_enabled,
            prompts_logger=self.prompts_logger,
            # Potentially pass lmstudio_api_key if FastAPIProxyThread is updated to use it
            # api_key=self.lmstudio_api_key
        )

        # Connect to signals from LlamaRunnerManager instance
        self.llama_runner_manager.runner_port_ready_for_proxy.connect(self.fastapi_proxy_thread.on_runner_port_ready)
        self.llama_runner_manager.runner_stopped_for_proxy.connect(self.fastapi_proxy_thread.on_runner_stopped)

        self.fastapi_proxy_thread.start()


    def start_ollama_proxy(self):
        """
        Starts the Ollama proxy in a separate thread.
        """
        if self.ollama_proxy_thread is not None and self.ollama_proxy_thread.isRunning():
            print("Ollama Proxy is already running.")
            return

        print("Starting Ollama Proxy...")

        self.ollama_proxy_thread = OllamaProxyThread(
            all_models_config=self.models,
            runtimes_config=self.llama_runtimes,
            is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
            get_runner_port_callback=self.llama_runner_manager.get_runner_port,
            request_runner_start_callback=self.llama_runner_manager.request_runner_start,
            prompt_logging_enabled=self.prompt_logging_enabled,
            prompts_logger=self.prompts_logger,
        )

        # Connect to signals from LlamaRunnerManager instance
        self.llama_runner_manager.runner_port_ready_for_proxy.connect(self.ollama_proxy_thread.on_runner_port_ready)
        self.llama_runner_manager.runner_stopped_for_proxy.connect(self.ollama_proxy_thread.on_runner_stopped)

        self.ollama_proxy_thread.start()


    def stop_fastapi_proxy(self):
        """
        Stops the FastAPI proxy thread.
        """
        if self.fastapi_proxy_thread and self.fastapi_proxy_thread.isRunning():
            print("Stopping FastAPI Proxy...")

            self.fastapi_proxy_thread.stop()

        else:
            print("FastAPI Proxy is not running.")


    def stop_ollama_proxy(self):
        """
        Stops the Ollama proxy thread.
        """
        if self.ollama_proxy_thread and self.ollama_proxy_thread.isRunning():
            print("Stopping Ollama Proxy...")

            self.ollama_proxy_thread.stop()
        else:
            print("Ollama Proxy is not running.")


    # Runner management slots moved to LlamaRunnerManager

    def customEvent(self, event: QEvent):
        """
        Handles custom events posted from other threads, primarily from LlamaRunnerManager.
        """
        # Check for events from LlamaRunnerManager (posted to its parent, which is MainWindow)
        if event.type() == LlamaRunnerManager.MANAGER_PARENT_NOTIFICATION_EVENT_TYPE:
            model_name = getattr(event, 'model_name', None)
            if model_name and hasattr(self, "llama_runner_manager"):
                # This event type from LlamaRunnerManager is used for cleanup/status updates
                # typically when a runner is found to be already stopped or exited unexpectedly.
                logging.debug(f"MainWindow received MANAGER_PARENT_NOTIFICATION_EVENT for {model_name}")
                self.llama_runner_manager.on_llama_runner_stopped(model_name)
            else:
                logging.warning("MainWindow received MANAGER_PARENT_NOTIFICATION_EVENT without model_name or llama_runner_manager not found.")

        else:
            super().customEvent(event)


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
