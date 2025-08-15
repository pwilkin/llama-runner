import asyncio
import subprocess
import logging
import re
import collections  # Import collections
from typing import Optional

from PySide6.QtCore import QObject, Signal

from llama_runner.config_loader import CONFIG_DIR, LOG_FILE

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaCppRunner(QObject):
    stopped = Signal()  # Define the stopped signal

    def __init__(
        self, model_name: str, model_path: str, llama_cpp_runtime: str = None, **kwargs # type: ignore
    ):
        """
        Initializes the LlamaCppRunner.

        Args:
            model_name (str): The display name of the model.
            model_path (str): The full path to the model file.
            llama_cpp_runtime (str, optional): The path to the llama-server executable.
                Defaults to None, which uses the llama-server from the PATH.
            **kwargs: Additional arguments to pass to llama-server.
        """
        super().__init__()  # Call QObject constructor
        self.model_name = model_name
        self.model_path = model_path
        self.llama_cpp_runtime = llama_cpp_runtime or "llama-server"
        self.kwargs = kwargs
        self.process: Optional[asyncio.subprocess.Process] = None
        self.startup_pattern = re.compile(
            r"main: server is listening on"
        )  # Regex to detect startup
        self.alt_startup_pattern = re.compile("HTTP server listening")  # ik.llama
        self.port = None  # Dynamically assigned port
        self._output_buffer = collections.deque(
            maxlen=10
        )  # Store the last 10 lines read from stdout

    async def start(self):
        """
        Starts the llama.cpp server.
        """
        if self.process and self.process.returncode is None:
            print(f"llama.cpp server for {self.model_name} is already running.")
            return

        command = [
            self.llama_cpp_runtime,
            "--model",
            self.model_path,
            "--alias",
            self.model_name,
            "--host",
            "127.0.0.1",  # Bind to localhost
            "--port",
            (
                "0" if "port" not in self.kwargs else str(self.kwargs["port"])
            ),  # Dynamically assigned port unless specified
        ]

        # Add additional arguments from kwargs
        for key, value in self.kwargs.items():
            if key == "port":
                continue  # already handled above
            arg_name = key.replace("_", "-")  # Convert snake_case to kebab-case
            # Handle boolean flags: if value is True, just add the flag, otherwise skip
            if isinstance(value, bool):
                if value:
                    command.append(f"--{arg_name}")
                # If value is False, do nothing (don't add --no-flag unless explicitly handled)
            else:
                command.append(f"--{arg_name}")
                command.append(str(value))

        logging.info(f"Starting llama.cpp server with command: {' '.join(command)}")

        # Clear the output buffer before starting a new process
        self._output_buffer.clear()

        try:
            # Use asyncio.create_subprocess_exec for better integration with asyncio
            # Need to capture stdout/stderr for reading
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                cwd=CONFIG_DIR,
            ) # type: ignore
            print(f"Process started with PID: {self.process.pid}")

            # Wait for startup message
            startup_success = await self.wait_for_server_startup()

            # If startup wasn't successful, check if the process exited immediately
            if not startup_success and self.process.returncode is not None:
                # Process exited before startup message
                # The error message will be generic, the UI will show the buffer
                raise RuntimeError(
                    f"Llama.cpp server for {self.model_name} exited during startup."
                )

            # If startup_success is False here, it means the process is running but didn't print the expected line
            # This case is handled in wait_for_server_startup by returning False

        except FileNotFoundError:
            error_msg = f"Error: Llama.cpp runtime not found at '{self.llama_cpp_runtime}'. Make sure it's in your PATH or the path is correct."
            print(error_msg)
            logging.error(error_msg)
            # Set process to None to indicate failure before process creation
            self.process = None # type: ignore
            raise RuntimeError(error_msg)  # Re-raise to be caught by the thread
        except Exception as e:
            error_msg = f"Error starting llama.cpp server process: {e}"
            print(error_msg)
            logging.error(error_msg)
            # Ensure process is None if startup failed before wait_for_server_startup
            if self.process and self.process.returncode is None:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=5) # type: ignore
                except:
                    self.process.kill()
                self.process = None # type: ignore
            elif self.process and self.process.returncode is not None:
                # Process started but exited quickly
                pass  # Handled above if startup_success is False
            else:
                self.process = None  # type: ignore # Ensure process is None on other errors
            raise RuntimeError(error_msg)  # Re-raise to be caught by the thread

    async def wait_for_server_startup(self):
        """
        Waits for the llama.cpp server to start up, using a regex pattern.
        Reads stdout line by line.
        """
        if not self.process or not self.process.stdout:
            print("Process or stdout not available to wait for startup.")
            return False

        print(f"Waiting for startup message for {self.model_name}...")
        try:
            while True:
                # Read a line with a timeout to prevent infinite blocking
                try:
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(), timeout=1.0
                    )  # Read line with timeout
                except asyncio.TimeoutError:
                    # Check if process is still running during timeout
                    if self.process.returncode is not None:
                        print(
                            f"Process for {self.model_name} exited while waiting for output."
                        )
                        return False  # Process exited

                    # If process is still running, continue waiting for output
                    continue  # Go back to the start of the while loop to try reading again

                if not line:
                    # End of stream reached, process likely exited
                    print(
                        f"End of stdout stream reached for {self.model_name}. Process likely exited."
                    )
                    return False

                decoded_line = line.decode("utf-8", errors="replace").strip()
                self._output_buffer.append(decoded_line)  # Store the line in the buffer

                print(f"llama.cpp[{self.model_name}]: {decoded_line}")

                match = self.startup_pattern.search(decoded_line)
                alt_match = self.alt_startup_pattern.search(decoded_line)
                if match or alt_match:
                    print(
                        f"llama.cpp server for {self.model_name} started successfully."
                    )
                    # Extract port from the startup message
                    if match:
                        port_match = re.search(
                            r"http://127\.0\.0\.1:(\d+)", decoded_line
                        )
                        if port_match:
                            self.port = int(port_match.group(1))
                            print(
                                f"llama.cpp server for {self.model_name} is listening on port {self.port}"
                            )
                        else:
                            # Startup line found, but port not extracted - this is also a failure
                            print(
                                f"Warning: Startup line found but port could not be extracted for {self.model_name}."
                            )
                            # The error message will be generic, the UI will show the buffer
                            return False  # Indicate failure if port isn't found
                    elif alt_match:
                        port_match = re.search(r'port="(\d+)"', decoded_line)
                        if port_match:
                            self.port = int(port_match.group(1))
                            print(
                                f"llama.cpp server for {self.model_name} is listening on port {self.port}"
                            )
                        else:
                            # Startup line found, but port not extracted - this is also a failure
                            print(
                                f"Warning: Startup line found but port could not be extracted for {self.model_name}."
                            )
                            # The error message will be generic, the UI will show the buffer
                            return False  # Indicate failure if port isn't found
                    return True  # Startup successful

                # Check if process exited after reading a line
                if self.process.returncode is not None:
                    print(f"Process for {self.model_name} exited after reading a line.")
                    return False  # Process exited

        except Exception as e:
            print(f"Error during startup wait for {self.model_name}: {e}")
            logging.error(f"Error during startup wait for {self.model_name}: {e}")
            # The error message will be generic, the UI will show the buffer
            return False  # Indicate failure

    async def stop(self):
        """
        Stops the llama.cpp server.
        """
        if self.process:
            logging.info(f"stop() called for {self.model_name}. PID: {self.process.pid}, Return Code: {self.process.returncode}")
        else:
            logging.info(f"stop() called for {self.model_name}. Process is None.")

        if (
            self.process and self.process.returncode is None
        ):  # Check if process is running
            print(
                f"Stopping llama.cpp server for {self.model_name} (PID: {self.process.pid})."
            )
            logging.info(f"Attempting to stop {self.model_name} (PID: {self.process.pid}). Current return code: {self.process.returncode}")
            try:
                logging.info(f"Calling terminate() on PID: {self.process.pid} for {self.model_name}")
                self.process.terminate()
                # Wait for the process to terminate gracefully
                await asyncio.wait_for(self.process.wait(), timeout=15) # type: ignore
                print(f"llama.cpp server for {self.model_name} terminated gracefully.")
                logging.info(f"PID: {self.process.pid} for {self.model_name} terminated gracefully.")
            except asyncio.TimeoutError:
                print(
                    f"llama.cpp server for {self.model_name} did not terminate in 15 seconds, killing it."
                )
                logging.warning(f"Timeout stopping PID: {self.process.pid} for {self.model_name}. Killing.")
                try:
                    logging.info(f"Calling kill() on PID: {self.process.pid} for {self.model_name}")
                    self.process.kill()
                    await asyncio.wait_for(
                        self.process.wait(), timeout=5 # type: ignore
                    )  # Wait a bit for kill
                    logging.info(f"PID: {self.process.pid} for {self.model_name} killed.")
                except Exception as kill_e:
                    print(f"Error killing process {self.process.pid}: {kill_e}")
                    logging.error(f"Error killing PID: {self.process.pid} for {self.model_name}: {kill_e}")
            except Exception as e:
                print(f"Error during process termination for {self.model_name}: {e}")
                logging.error(f"Exception during termination of PID: {self.process.pid} for {self.model_name}: {e}")
            finally:
                # Emit a signal to notify the GUI that the server has stopped
                logging.info(f"Emitting stopped signal for {self.model_name} (PID: {self.process.pid if self.process else 'N/A'}). Final return code: {self.process.returncode if self.process else 'N/A'}")
                self.stopped.emit()
        elif self.process and self.process.returncode is not None:
            print(
                f"llama.cpp server for {self.model_name} was already stopped (code {self.process.returncode})."
            )
            logging.info(f"stop() called for {self.model_name} (PID: {self.process.pid}), but process already stopped with code {self.process.returncode}.")
            self.process = None  # type: ignore # Ensure process reference is cleared
        else:
            print(
                f"llama.cpp server for {self.model_name} is not running (no process)."
            )
            logging.info(f"stop() called for {self.model_name}, but no process was running.")
            self.process = None  # type: ignore # Ensure process reference is cleared

    def is_running(self):
        """
        Checks if the llama.cpp server process is running.
        """
        return self.process is not None and self.process.returncode is None

    def get_port(self):
        """
        Returns the port number of the llama.cpp server.
        """
        return self.port

    def get_last_output_line(self):
        """
        Returns the last line read from the process's stdout during startup wait.
        (Deprecated, use get_output_buffer instead)
        """
        return self._output_buffer[-1] if self._output_buffer else None

    def get_output_buffer(self):
        """
        Returns the list of lines currently in the output buffer.
        """
        return list(self._output_buffer)
