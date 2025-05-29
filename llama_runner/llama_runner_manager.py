import asyncio
import os
import logging
from typing import Optional, Dict, Any

from PySide6.QtCore import QObject, QTimer, QCoreApplication, QEvent, Signal, Slot

# Import the custom event classes
from llama_runner.llama_runner_thread import LlamaRunnerThread, RunnerStoppedEvent, RunnerErrorEvent
from llama_runner.error_output_dialog import ErrorOutputDialog

class LlamaRunnerManager(QObject):
    # Define a custom event type for events posted to the parent (e.g., MainWindow)
    # This replaces the QEvent.User + 4 magic number.
    MANAGER_PARENT_NOTIFICATION_EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    # Define signals directly as class attributes
    runner_port_ready_for_proxy = Signal(str, int)
    runner_stopped_for_proxy = Signal(str)

    def __init__(
        self,
        models: dict,
        llama_runtimes: dict,
        default_runtime: str,
        model_status_widgets: dict,
        # runner_port_ready_for_proxy and runner_stopped_for_proxy are now class attributes
        parent=None,
    ):
        super().__init__(parent)
        # Import here to avoid circular import issues
        # self.LlamaRunnerThread = LlamaRunnerThread # LlamaRunnerThread is already imported directly
        self.models = models
        self.llama_runtimes = llama_runtimes
        self.default_runtime = default_runtime
        self.model_status_widgets = model_status_widgets
        # self.runner_port_ready_for_proxy = runner_port_ready_for_proxy # Removed, now class attribute
        # self.runner_stopped_for_proxy = runner_stopped_for_proxy # Removed, now class attribute

        self.llama_runner_threads: Dict[str, LlamaRunnerThread] = {}
        self._runner_startup_futures: Dict[str, asyncio.Future] = {}
        self._current_running_model: Optional[str] = None
        self.concurrent_runners_limit = 1  # Will be set by MainWindow after instantiation

    def set_concurrent_runners_limit(self, limit: int):
        self.concurrent_runners_limit = limit

    def is_llama_runner_running(self, model_name: str) -> bool:
        thread = self.llama_runner_threads.get(model_name)
        if thread and thread.isRunning() and thread.runner and thread.runner.is_running():
            return True
        return False

    def get_runner_port(self, model_name: str) -> Optional[int]:
        thread = self.llama_runner_threads.get(model_name)
        if thread and thread.isRunning() and thread.runner and thread.runner.is_running():
            return thread.runner.get_port()
        return None

    def request_runner_start(self, model_name: str) -> asyncio.Future:
        logging.info(f"Received request to start runner for model: {model_name}")

        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
            logging.info(f"Runner for {model_name} is already starting. Returning existing Future.")
            return self._runner_startup_futures[model_name]

        if self.is_llama_runner_running(model_name):
            port = self.get_runner_port(model_name)
            if port is not None:
                logging.info(f"Runner for {model_name} is already running on port {port}. Returning completed Future.")
                future = asyncio.Future()
                future.set_result(port)
                self._runner_startup_futures[model_name] = future
                QTimer.singleShot(1000, lambda: self._cleanup_completed_future(model_name))
                return future
            else:
                logging.error(f"Runner for {model_name} is reported as running but port is None.")
                future = asyncio.Future()
                future.set_exception(RuntimeError(f"Runner for {model_name} is running but port is unavailable."))
                self._runner_startup_futures[model_name] = future
                QTimer.singleShot(1000, lambda: self._cleanup_completed_future(model_name))
                return future

        running_runners = {name: thread for name, thread in self.llama_runner_threads.items() if thread.isRunning()}
        num_running = len(running_runners)

        if num_running >= self.concurrent_runners_limit:
            if self.concurrent_runners_limit == 1:
                models_to_stop = list(running_runners.keys())
                if models_to_stop:
                    logging.info(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Stopping existing runner(s): {models_to_stop} before starting {model_name}.")
                    for name_to_stop in models_to_stop:
                        self.stop_llama_runner(name_to_stop)
                else:
                    logging.warning("Concurrent runner limit reached but no running runners found?")
            else:
                logging.warning(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Cannot start {model_name}.")
                future = asyncio.Future()
                future.set_exception(RuntimeError(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Cannot start '{model_name}'."))
                self._runner_startup_futures[model_name] = future
                QTimer.singleShot(1000, lambda: self._cleanup_completed_future(model_name))
                return future

        future = asyncio.Future()
        self._runner_startup_futures[model_name] = future

        model_config = self.models[model_name]
        model_path = model_config.get("model_path")
        llama_cpp_runtime_key = model_config.get("llama_cpp_runtime", "default")
        _raw_llama_cpp_runtime_config = self.llama_runtimes.get(llama_cpp_runtime_key, self.default_runtime)

        if isinstance(_raw_llama_cpp_runtime_config, dict):
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config.get("runtime")
            if not llama_cpp_runtime_command:
                logging.error(f"Runtime configuration for '{llama_cpp_runtime_key}' is a dict but missing 'runtime' key.")
                future.set_exception(RuntimeError(f"Invalid runtime config for '{llama_cpp_runtime_key}'."))
                return future
        elif isinstance(_raw_llama_cpp_runtime_config, str):
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config
        else:
            logging.error(f"Unexpected type for runtime configuration '{llama_cpp_runtime_key}': {type(_raw_llama_cpp_runtime_config)}")
            future.set_exception(RuntimeError(f"Invalid runtime configuration type for '{llama_cpp_runtime_key}'."))
            return future

        if not model_path:
            logging.error(f"Configuration Error: Model '{model_name}' has no 'model_path' specified in config.json.")
            future.set_exception(RuntimeError(f"Configuration Error: Model '{model_name}' has no 'model_path'."))
            return future

        if not os.path.exists(model_path):
            logging.error(f"File Not Found: Model file not found: {model_path}")
            future.set_exception(FileNotFoundError(f"Model file not found: {model_path}"))
            return future

        if llama_cpp_runtime_key != "default" and not os.path.exists(llama_cpp_runtime_command):
            logging.error(f"Runtime Not Found: Llama.cpp runtime not found: {llama_cpp_runtime_command}")
            future.set_exception(FileNotFoundError(f"Llama.cpp runtime not found: {llama_cpp_runtime_command}"))
            return future

        print(f"Starting Llama Runner for {model_name}...")
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_status("Starting...")
            status_widget.update_port("N/A")
            status_widget.set_buttons_enabled(False, False)

        thread = LlamaRunnerThread(
            model_name=model_name,
            model_path=model_path,
            llama_cpp_runtime=llama_cpp_runtime_command,
            parent=self,
            **model_config.get("parameters", {})
        )
        thread.started.connect(lambda name=model_name: self.on_llama_runner_started(name))
        thread.port_ready.connect(self.on_llama_runner_port_ready_and_emit)
        thread.error.connect(lambda message, output_buffer, name=model_name: self.on_llama_runner_error(name, message, output_buffer))
        # thread.stopped.connect(lambda name=model_name: self.on_llama_runner_stopped(name))  # Removed, now handled by customEvent

        self.llama_runner_threads[model_name] = thread
        thread.start()

        return future

    def _cleanup_completed_future(self, model_name: str):
        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
            logging.debug(f"Cleaning up completed future for {model_name}")
            del self._runner_startup_futures[model_name]

    def stop_llama_runner(self, model_name: str):
        if model_name in self.llama_runner_threads and self.llama_runner_threads[model_name].isRunning():
            print(f"Stopping Llama Runner for {model_name}...")
            status_widget = self.model_status_widgets.get(model_name)
            if status_widget:
                status_widget.update_status("Stopping...")
                status_widget.set_buttons_enabled(False, False)
            self.llama_runner_threads[model_name].stop()
        else:
            logging.warning(f"Attempted to stop non-running thread {model_name}. Cleaning up state.")
            # If the thread isn't running, we might still want to ensure UI cleanup.
            # The original code posted an event to self.parent().
            # This could be a signal to MainWindow or similar.
            # We'll keep this logic for now, assuming it's handled by the parent.
            # However, if this is meant to trigger on_llama_runner_stopped,
            # it should use RunnerStoppedEvent and post to self.
            if model_name in self.llama_runner_threads: # Check if thread object exists
                # To trigger on_llama_runner_stopped for cleanup if thread is gone but not cleaned:
                # self.on_llama_runner_stopped(model_name)
                # Or, if an event is preferred for the manager itself:
                # QCoreApplication.instance().postEvent(self, RunnerStoppedEvent(model_name))

                # Original logic: post a generic event to parent.
                # This might be for a different purpose than the thread's stopped event.
                parent_event = QEvent(LlamaRunnerManager.MANAGER_PARENT_NOTIFICATION_EVENT_TYPE)
                setattr(parent_event, 'model_name', model_name) # Dynamically add attr if needed by parent
                app_instance = QCoreApplication.instance()
                parent_object = self.parent()
                if app_instance and parent_object:
                    app_instance.postEvent(parent_object, parent_event)
                else:
                    logging.warning(f"Could not post parent event for {model_name} (stop non-running): App/Parent None.")
            else: # If thread object doesn't even exist, call stop handler directly for cleanup
                self.on_llama_runner_stopped(model_name)


    def stop_all_llama_runners(self):
        print("Stopping all Llama Runners...")
        # Collect running threads first to avoid modifying the dict during iteration
        running_threads = [
            (model_name, thread)
            for model_name, thread in self.llama_runner_threads.items()
            if thread.isRunning()
        ]
        for model_name, thread in running_threads:
            self.stop_llama_runner(model_name)
        # Wait for all threads to finish
        for model_name, thread in running_threads:
            thread.wait()

    @Slot(str)
    def on_llama_runner_started(self, model_name: str):
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_status("Starting...")
            status_widget.set_buttons_enabled(False, False)

    @Slot(str)
    def on_llama_runner_stopped(self, model_name: str):
        print(f"Llama Runner for {model_name} stopped.")
        if model_name in self.llama_runner_threads:
            thread = self.llama_runner_threads.pop(model_name)
            thread.deleteLater()
            status_widget = self.model_status_widgets.get(model_name)
            if status_widget:
                status_widget.update_status("Not Running")
                status_widget.update_port("N/A")
                status_widget.set_buttons_enabled(True, False)
            if self._current_running_model == model_name:
                self._current_running_model = None
                logging.info(f"Cleared current running model: {model_name}")
            self.runner_stopped_for_proxy.emit(model_name)
            if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
                logging.debug(f"Runner {model_name} stopped unexpectedly while startup Future was pending.")
                self._runner_startup_futures[model_name].set_exception(RuntimeError(f"Runner for {model_name} stopped unexpectedly during startup."))
        else:
            logging.warning(f"Stopped signal received for unknown or already cleaned up model: {model_name}")

    @Slot(str, str, list)
    def on_llama_runner_error(self, model_name: str, message: str, output_buffer: list):
        print(f"Llama Runner for {model_name} error: {message}")
        dialog_message = f"Llama.cpp server for {model_name} encountered an error:\n{message}"
        error_dialog = ErrorOutputDialog(
            title=f"Llama Runner Error: {model_name}",
            message=dialog_message,
            output_lines=output_buffer,
            parent=self.parent()
        )
        error_dialog.exec()
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_status("Error")
        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
            logging.debug(f"Runner {model_name} errored while startup Future was pending.")
            self._runner_startup_futures[model_name].set_exception(RuntimeError(f"Runner for {model_name} errored during startup: {message}"))

    def customEvent(self, event: QEvent):
        # Handle custom stopped event from LlamaRunnerThread
        if event.type() == RunnerStoppedEvent.EVENT_TYPE:
            if isinstance(event, RunnerStoppedEvent):
                self.on_llama_runner_stopped(event.model_name)
            else:
                logging.warning(f"Received RunnerStoppedEvent.EVENT_TYPE but event is not RunnerStoppedEvent instance: {type(event)}")
        elif event.type() == RunnerErrorEvent.EVENT_TYPE:
            if isinstance(event, RunnerErrorEvent):
                self.on_llama_runner_error(event.model_name, event.message, event.output_buffer)
            else:
                logging.warning(f"Received RunnerErrorEvent.EVENT_TYPE but event is not RunnerErrorEvent instance: {type(event)}")
        else:
            super().customEvent(event)

    @Slot(str, int)
    def on_llama_runner_port_ready_and_emit(self, model_name: str, port: int):
        print(f"Llama Runner for {model_name} ready on port {port}.")
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_port(port)
            status_widget.update_status("Running")
            status_widget.set_buttons_enabled(False, True)
        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
            logging.debug(f"Resolving runner_startup_future for {model_name} with port {port}")
            self._runner_startup_futures[model_name].set_result(port)
        elif model_name in self._runner_startup_futures and self._runner_startup_futures[model_name].done():
            logging.warning(f"Runner_port_ready signal received for {model_name}, but Future was already done.")
        else:
            logging.warning(f"Runner_port_ready signal received for {model_name}, but no pending Future found.")
        self._current_running_model = model_name
        logging.info(f"Set current running model: {model_name}")
        self.runner_port_ready_for_proxy.emit(model_name, port)

    @Slot()
    def check_runner_statuses(self):
        for model_name, thread in list(self.llama_runner_threads.items()):
            if thread.isRunning() and thread.runner and thread.runner.process:
                return_code = thread.runner.process.returncode
                if return_code is not None:
                    logging.warning(f"Detected exited runner process for {model_name}. Return code: {return_code}")
                    # This posts an event to the parent, similar to stop_llama_runner for non-running.
                    parent_event = QEvent(LlamaRunnerManager.MANAGER_PARENT_NOTIFICATION_EVENT_TYPE)
                    setattr(parent_event, 'model_name', model_name) # Dynamically add attr if needed by parent
                    app_instance = QCoreApplication.instance()
                    parent_object = self.parent()
                    if app_instance and parent_object:
                        app_instance.postEvent(parent_object, parent_event)
                    else:
                        logging.warning(f"Could not post parent event for {model_name} (check_runner_statuses): App/Parent None.")