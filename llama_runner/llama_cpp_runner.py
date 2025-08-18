import asyncio
import logging
import re
import collections
import signal
from typing import Optional, Callable, List

from llama_runner.config_loader import CONFIG_DIR, LOG_FILE

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class LlamaCppRunner:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        llama_cpp_runtime: str = None,
        on_started: Callable[[str], None] = None,
        on_stopped: Callable[[str], None] = None,
        on_error: Callable[[str, str, List[str]], None] = None,
        on_port_ready: Callable[[str, int], None] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.llama_cpp_runtime = llama_cpp_runtime or "llama-server"
        self.on_started = on_started
        self.on_stopped = on_stopped
        self.on_error = on_error
        self.on_port_ready = on_port_ready
        self.kwargs = kwargs
        self.process: Optional[asyncio.subprocess.Process] = None
        self.startup_pattern = re.compile(r"main: server is listening on")
        self.alt_startup_pattern = re.compile("HTTP server listening")
        self.port = None
        self._output_buffer = collections.deque(maxlen=200) # Increased buffer size
        self._is_stopping = False

    async def _read_output_continuously(self, stream):
        while True:
            try:
                line = await stream.readline()
                if not line:
                    break
                decoded_line = line.decode("utf-8", errors="replace").strip()
                self._output_buffer.append(decoded_line)
                logging.debug(f"llama.cpp[{self.model_name}]: {decoded_line}")

                if self.port is None:
                    match = self.startup_pattern.search(decoded_line)
                    alt_match = self.alt_startup_pattern.search(decoded_line)
                    if match or alt_match:
                        port_match = None
                        if match:
                            port_match = re.search(r"http://127\.0\.0\.1:(\d+)", decoded_line)
                        elif alt_match:
                            port_match = re.search(r'port="(\d+)"', decoded_line)

                        if port_match:
                            self.port = int(port_match.group(1))
                            logging.info(f"llama.cpp server for {self.model_name} is listening on port {self.port}")
                            if self.on_port_ready:
                                self.on_port_ready(self.model_name, self.port)
                        else:
                            logging.warning(f"Startup line found but port could not be extracted for {self.model_name}.")
            except asyncio.CancelledError:
                logging.debug(f"Log reader for {self.model_name} cancelled.")
                break
            except Exception as e:
                logging.error(f"Error reading output for {self.model_name}: {e}")
                break

    async def run(self):
        log_reader_task = None
        try:
            if self.on_started:
                self.on_started(self.model_name)

            await self.start()

            if self.process and self.process.stdout:
                log_reader_task = asyncio.create_task(self._read_output_continuously(self.process.stdout))
            else:
                raise RuntimeError("Process or stdout not available after start.")

            return_code = await self.process.wait()
            logging.info(f"Process for {self.model_name} exited with code {return_code}.")

            # If the process was told to stop, and it exited with SIGTERM, that's not an error.
            if self._is_stopping and return_code == -signal.SIGTERM:
                logging.info(f"Llama.cpp server for {self.model_name} was stopped gracefully.")
            elif return_code != 0:
                error_msg = f"Llama.cpp server for {self.model_name} exited unexpectedly with code {return_code}."
                logging.error(error_msg)
                if self.on_error:
                    self.on_error(self.model_name, error_msg, self.get_output_buffer())

        except Exception as e:
            error_msg = f"Error running llama.cpp server: {e}"
            logging.error(error_msg, exc_info=True)
            if self.on_error:
                self.on_error(self.model_name, error_msg, self.get_output_buffer())
        finally:
            if log_reader_task and not log_reader_task.done():
                log_reader_task.cancel()
                await log_reader_task
            if self.on_stopped:
                self.on_stopped(self.model_name)

    async def start(self):
        if self.process and self.process.returncode is None:
            logging.warning(f"llama.cpp server for {self.model_name} is already running.")
            return

        command = [
            self.llama_cpp_runtime, "--model", self.model_path, "--alias", self.model_name,
            "--host", "127.0.0.1", "--port", "0" if "port" not in self.kwargs else str(self.kwargs["port"]),
        ]
        for key, value in self.kwargs.items():
            if key == "port":
                continue
            arg_name = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(f"--{arg_name}")
            else:
                command.extend([f"--{arg_name}", str(value)])

        logging.info(f"Starting llama.cpp server with command: {' '.join(command)}")
        self._output_buffer.clear()

        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=CONFIG_DIR,
            )
            logging.info(f"Process started with PID: {self.process.pid}")
        except FileNotFoundError:
            error_msg = f"Error: Llama.cpp runtime not found at '{self.llama_cpp_runtime}'."
            logging.error(error_msg)
            self.process = None
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error starting llama.cpp server process: {e}"
            logging.error(error_msg)
            self.process = None
            raise RuntimeError(error_msg)

    async def stop(self):
        if not self.process or self.process.returncode is not None:
            logging.info(f"stop() called for {self.model_name}, but process was not running.")
            return

        self._is_stopping = True

        logging.info(f"Stopping {self.model_name} (PID: {self.process.pid}).")
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=15)
            logging.info(f"PID: {self.process.pid} for {self.model_name} terminated gracefully.")
        except asyncio.TimeoutError:
            logging.warning(f"Timeout stopping PID: {self.process.pid} for {self.model_name}. Killing.")
            self.process.kill()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
                logging.info(f"PID: {self.process.pid} for {self.model_name} killed.")
            except Exception as kill_e:
                logging.error(f"Error killing PID: {self.process.pid} for {self.model_name}: {kill_e}")
        except Exception as e:
            logging.error(f"Exception during termination of PID: {self.process.pid} for {self.model_name}: {e}")

    def is_running(self):
        return self.process is not None and self.process.returncode is None

    def get_port(self):
        return self.port

    def get_output_buffer(self):
        return list(self._output_buffer)
