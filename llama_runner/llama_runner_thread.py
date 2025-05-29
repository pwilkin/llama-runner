import asyncio
import logging
import traceback
from typing import Optional, List

from PySide6.QtCore import QThread, Signal, QEvent, QCoreApplication

from llama_runner.llama_cpp_runner import LlamaCppRunner

# Custom Event Types
# It's generally safer to register event types to avoid conflicts.
# If QEvent.User is problematic with Pylance, direct integers or registered types are better.
# For simplicity and given the existing QEvent.User usage, we'll try to keep it,
# but subclassing QEvent is the most robust way to pass data.

# Use QEvent.registerEventType() for unique event IDs.
# It's important that this is called only once per type.
# We can do this by assigning the result to a class variable.

class RunnerStoppedEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    def __init__(self, model_name: str):
        super().__init__(RunnerStoppedEvent.EVENT_TYPE)
        self.model_name = model_name

class RunnerErrorEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    def __init__(self, model_name: str, message: str, output_buffer: List[str]):
        super().__init__(RunnerErrorEvent.EVENT_TYPE)
        self.model_name = model_name
        self.message = message
        self.output_buffer = output_buffer


class LlamaRunnerThread(QThread):
    """
    QThread to run the LlamaCppRunner in a separate thread to avoid blocking the UI.
    """
    # Define a custom event type for signaling that the runner has stopped
    # STOPPED_EVENT_TYPE = QEvent.Type(QEvent.User + 5) # Replaced by RunnerStoppedEvent
    # ERROR_EVENT_TYPE = QEvent.Type(QEvent.User + 3)  # Replaced by RunnerErrorEvent

    started = Signal(str)
    # stopped = Signal(str) # Removed direct stopped signal
    error = Signal(str, list) # Signal includes error message and output buffer
    port_ready = Signal(str, int) # Signal includes model_name and port

    def __init__(self, model_name: str, model_path: str, llama_cpp_runtime: Optional[str] = None, parent=None, **kwargs):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.llama_cpp_runtime = llama_cpp_runtime
        # Remove 'parent' from kwargs if present to avoid passing it to LlamaCppRunner
        if 'parent' in kwargs:
            del kwargs['parent']
        self.kwargs = kwargs
        self.runner = None
        self.is_running = False
        self._error_emitted = False # Flag to track if error signal was emitted
        self._manual_stop_requested = False # Flag to indicate if stop was manual
        self._output_reader_task: Optional[asyncio.Task] = None # Task for continuous output reading

    def run(self):
        """
        Runs the LlamaCppRunner in the thread.
        """
        self.is_running = True
        self._error_emitted = False # Reset flag
        self._manual_stop_requested = False # Reset flag
        self._output_reader_task = None # Reset task reference

        try:
            # Use a new event loop for this thread
            # Note: This thread runs its own asyncio loop. Bridging is needed
            # to communicate with the main thread's Qt loop.
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            # This catch is mostly for unexpected errors in the asyncio loop itself
            logging.error(f"Unexpected error in LlamaRunnerThread run: {e}\n{traceback.format_exc()}")
            # If an error occurred here and not in run_async, emit a generic error
            if not self._error_emitted:
                # Pass an empty list for output buffer in this case
                self.error.emit(f"Unexpected thread error: {e}", [])
                self._error_emitted = True
        finally:
            # Ensure the output reader task is cancelled if it's running
            if self._output_reader_task and not self._output_reader_task.done():
                try:
                    self._output_reader_task.cancel()
                    # Wait briefly for cancellation to complete
                    self.loop.run_until_complete(asyncio.gather(self._output_reader_task, return_exceptions=True))
                except Exception as cancel_e:
                    logging.error(f"Error cancelling output reader task: {cancel_e}")

            self.is_running = False
            # The stopped signal is emitted in run_async's finally block
            if hasattr(self, 'loop') and self.loop.is_running():
                self.loop.stop()
            if hasattr(self, 'loop') and not self.loop.is_closed():
                self.loop.close()

    async def run_async(self):
        """
        Asynchronous part of the runner.
        """
        try:
            self.runner = LlamaCppRunner(
                model_name=self.model_name,
                model_path=self.model_path,
                llama_cpp_runtime=self.llama_cpp_runtime if self.llama_cpp_runtime is not None else "", # Provide default for LlamaCppRunner
                **self.kwargs
            )
            # The start method now raises exceptions on failure
            await self.runner.start()

            # Check if startup message was found and port was set
            if self.runner.port is None:
                raise RuntimeError("Llama.cpp server failed to start or extract port.")

            # Emit signals after successful startup and port detection using Qt signals
            self.started.emit(self.model_name)
            self.port_ready.emit(self.model_name, self.runner.get_port())

            # Start continuous output reading task
            self._output_reader_task = self.loop.create_task(self._read_output_continuously())

            # Wait for the process to exit
            # If self.runner.process.wait() is blocking, run it in an executor
            if self.runner and self.runner.process:
                await self.runner.process.wait() # Correctly await the coroutine

            logging.info(f"Llama.cpp process for {self.model_name} exited with code {self.runner.process.returncode if self.runner and self.runner.process else 'N/A'}")

        except Exception as e:
            logging.error(f"Error running LlamaCppRunner: {e}\n{traceback.format_exc()}")
            # Emit the error message and the output buffer
            output_buffer = self.runner.get_output_buffer() if self.runner else []
            self.error.emit(str(e), output_buffer)
            self._error_emitted = True # Set flag

        finally:
            # Ensure the runner process is stopped if it's still running (shouldn't be if wait() completed)
            if self.runner and self.runner.is_running():
                logging.warning(f"Llama.cpp process for {self.model_name} was still running in finally block after wait(), stopping.")
                try:
                    # Use run_until_complete to await stop in the thread's loop
                    await self.runner.stop()
                except Exception as stop_e:
                    logging.error(f"Error stopping LlamaCppRunner in finally: {stop_e}\n{traceback.format_exc()}")
                    # Don't overwrite the main error_occurred flag

            # Check the final return code after ensuring the process is stopped
            # Only emit error if one hasn't been emitted by the except block AND it wasn't a manual stop
            # The process exit code is checked here after wait() returns
            if not self._error_emitted and not self._manual_stop_requested and self.runner and self.runner.process and self.runner.process.returncode != 0:
                error_message = f"Llama.cpp server for {self.model_name} exited with code {self.runner.process.returncode}"
                output_buffer = self.runner.get_output_buffer() if self.runner else []
                logging.error(error_message)
                # Use QCoreApplication.instance().postEvent to emit signals from the asyncio thread
                error_event = RunnerErrorEvent(self.model_name, error_message, output_buffer)
                app_instance = QCoreApplication.instance()
                parent_object = self.parent()
                logging.info(f"LRT ({self.model_name}): Attempting to post RunnerErrorEvent. Parent type: {type(parent_object)}, Parent exists: {bool(parent_object)}")
                if app_instance and parent_object:
                    logging.info(f"LRT ({self.model_name}): Posting RunnerErrorEvent to parent {parent_object}.")
                    app_instance.postEvent(parent_object, error_event)
                else:
                    logging.error(f"LRT ({self.model_name}): Could not post RunnerErrorEvent. App instance: {bool(app_instance)}, Parent: {bool(parent_object)}.")
                self._error_emitted = True # Set flag

            self.is_running = False # Ensure this is false
            # self.stopped.emit(self.model_name) # Removed direct stopped signal

            # Post a custom event to the parent (LlamaRunnerManager) to signal stopped
            stopped_event = RunnerStoppedEvent(self.model_name)
            app_instance = QCoreApplication.instance()
            parent_object = self.parent()
            logging.info(f"LRT ({self.model_name}): Attempting to post RunnerStoppedEvent. Parent type: {type(parent_object)}, Parent exists: {bool(parent_object)}")
            if app_instance and parent_object:
                logging.info(f"LRT ({self.model_name}): Posting RunnerStoppedEvent to parent {parent_object}.")
                app_instance.postEvent(parent_object, stopped_event)
            else:
                logging.error(f"LRT ({self.model_name}): Could not post RunnerStoppedEvent. App instance: {bool(app_instance)}, Parent: {bool(parent_object)}.")

    async def _read_output_continuously(self):
        """
        Reads output from the runner process stdout continuously after startup.
        """
        if not self.runner or not self.runner.process or not self.runner.process.stdout:
            logging.warning("Output stream not available for continuous reading.")
            return

        logging.info(f"Starting continuous output reading for {self.model_name}")
        try:
            while True:
                line = await self.runner.process.stdout.readline()
                if not line:
                    # End of stream reached, process likely exited
                    logging.info(f"End of stdout stream reached for {self.model_name}. Stopping continuous reading.")
                    break
                decoded_line = line.decode('utf-8', errors='replace').strip()
                # Append to the buffer (LlamaCppRunner already does this during startup wait)
                # We could add a separate buffer here if needed, but for now just log
                # self.runner._output_buffer.append(decoded_line) # Optionally append to buffer
                logging.info(f"llama.cpp[{self.model_name}]: {decoded_line}")

        except asyncio.CancelledError:
            logging.info(f"Output reader task for {self.model_name} cancelled.")
        except Exception as e:
            logging.error(f"Error during continuous output reading for {self.model_name}: {e}\n{traceback.format_exc()}")

    def stop(self):
        """
        Signals the LlamaCppRunner thread to stop.
        The actual stopping happens in run_async.
        """
        self.is_running = False
        self._manual_stop_requested = True # Set flag for manual stop

        # Cancel the output reader task if it's running
        if self._output_reader_task and not self._output_reader_task.done():
            self._output_reader_task.cancel()

        # If the asyncio loop is running, schedule the stop coroutine
        if hasattr(self, 'loop') and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._request_stop_runner(), self.loop)
        else:
            logging.warning(f"Attempted to stop runner {self.model_name} but loop was not running.")

    async def _request_stop_runner(self):
        """Helper coroutine to request runner stop from within the thread's loop."""
        if self.runner:
            await self.runner.stop()