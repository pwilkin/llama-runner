import os
import sys
import subprocess
import asyncio
from typing import Optional, Dict, List

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QStackedWidget)
from PySide6.QtCore import Slot, Signal, QCoreApplication

from llama_runner.config_loader import load_config
from llama_runner.lmstudio_proxy_thread import LMStudioProxyServer
from llama_runner.ollama_proxy_thread import OllamaProxyServer
from llama_runner.model_status_widget import ModelStatusWidget
from llama_runner.llama_runner_manager import LlamaRunnerManager
from llama_runner.error_output_dialog import ErrorOutputDialog

class MainWindow(QWidget):
    runner_port_ready_for_proxy = Signal(str, int)
    runner_stopped_for_proxy = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama Runner")
        self.resize(800, 600)
        self.config = load_config()
        
        self.llama_runtimes = self.config.get("llama-runtimes", {})
        self.default_runtime = self.config.get("default_runtime", "llama-server")
        self.models = self.config.get("models", {})
        self.concurrent_runners_limit = self.config.get("concurrentRunners", 1)

        proxies_config = self.config.get('proxies', {})
        self.ollama_proxy_enabled = proxies_config.get('ollama', {}).get('enabled', True)
        self.lmstudio_proxy_enabled = proxies_config.get('lmstudio', {}).get('enabled', True)

        self.model_status_widgets: Dict[str, ModelStatusWidget] = {}
        self.lmstudio_proxy_server: Optional[LMStudioProxyServer] = None
        self.ollama_proxy_server: Optional[OllamaProxyServer] = None

        self._setup_ui()

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

        if self.lmstudio_proxy_server:
            self.lmstudio_proxy_server.task = asyncio.create_task(self.lmstudio_proxy_server.start())
        if self.ollama_proxy_server:
            self.ollama_proxy_server.task = asyncio.create_task(self.ollama_proxy_server.start())

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.top_layout = QHBoxLayout()
        self.model_list_widget = QListWidget()
        self.model_list_widget.setMinimumWidth(150)
        self.top_layout.addWidget(self.model_list_widget)
        self.model_status_stack = QStackedWidget()
        self.top_layout.addWidget(self.model_status_stack)
        self.main_layout.addLayout(self.top_layout)

        for model_name in self.models.keys():
            self.model_list_widget.addItem(model_name)
            status_widget = ModelStatusWidget(model_name)
            self.model_status_stack.addWidget(status_widget)
            self.model_status_widgets[model_name] = status_widget
            status_widget.start_button.clicked.connect(lambda checked, name=model_name: asyncio.create_task(self.llama_runner_manager.request_runner_start(name)))
            status_widget.stop_button.clicked.connect(lambda checked, name=model_name: self.llama_runner_manager.stop_llama_runner(name))

        self.model_list_widget.currentItemChanged.connect(self.on_model_selection_changed)
        self.edit_config_button = QPushButton("Edit config")
        self.main_layout.addWidget(self.edit_config_button)
        self.edit_config_button.clicked.connect(self.open_config_file)

    def closeEvent(self, event):
        print("MainWindow closing. Stopping all services...")
        asyncio.create_task(self.stop_all_services())
        event.accept()

    async def stop_all_services(self):
        tasks_to_cancel = []
        if self.ollama_proxy_server:
            self.ollama_proxy_server.stop()
            tasks_to_cancel.append(self.ollama_proxy_server.task)
        if self.lmstudio_proxy_server:
            self.lmstudio_proxy_server.stop()
            tasks_to_cancel.append(self.lmstudio_proxy_server.task)

        await self.llama_runner_manager.stop_all_llama_runners_async()

        if tasks_to_cancel:
            await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)

        QCoreApplication.quit()

    @Slot(str)
    def on_model_selection_changed(self, current_item, previous_item):
        if current_item:
            self.model_status_stack.setCurrentWidget(self.model_status_widgets[current_item.text()])

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

    def on_runner_error(self, model_name: str, message: str, output_buffer: List[str]):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_status("Error")
        dialog = ErrorOutputDialog(
            title=f"Llama Runner Error: {model_name}",
            message=f"Llama.cpp server for {model_name} encountered an error:\n{message}",
            output_lines=output_buffer,
            parent=self
        )
        dialog.exec()

    def on_runner_port_ready(self, model_name: str, port: int):
        widget = self.model_status_widgets.get(model_name)
        if widget:
            widget.update_port(str(port))
            widget.update_status("Running")
            widget.set_buttons_enabled(False, True)
        self.runner_port_ready_for_proxy.emit(model_name, port)

    def open_config_file(self):
        config_path = os.path.expanduser("~/.llama-runner/config.json")
        if sys.platform == "win32":
            os.startfile(config_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", config_path])
        else:
            subprocess.run(["xdg-open", config_path])
