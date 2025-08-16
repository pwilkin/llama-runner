import asyncio
import logging
import traceback
from typing import Optional, List
from threading import Thread
from dataclasses import dataclass

from llama_runner.llama_cpp_runner import LlamaCppRunner

# Define dataclasses for events
@dataclass
class PortReadyEvent:
    model_name: str
    port: int

@dataclass
class RunnerStoppedEvent:
    model_name: str

@dataclass
class RunnerErrorEvent:
    model_name: str
    message: str
    output_buffer: List[str]

@dataclass
class RunnerStartedEvent:
    model_name: str


class LlamaRunnerThread(Thread):
    """
    A thread to run the LlamaCppRunner to avoid blocking.
    Communicates with the manager via an asyncio.Queue.
    """
    def __init__(self, model_name: str, model_path: str, event_queue: asyncio.Queue, llama_cpp_runtime: Optional[str] = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.event_queue = event_queue
        self.llama_cpp_runtime = llama_cpp_runtime
        self.kwargs = kwargs
        self.runner: Optional[LlamaCppRunner] = None
        self.is_running = False
        self._manual_stop_requested = False
        self._output_reader_task: Optional[asyncio.Task] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self):
        """
        Runs the LlamaCppRunner in the thread.
        """
        self.is_running = True
        self._manual_stop_requested = False
        self._output_reader_task = None

        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            logging.error(f"Unexpected error in LlamaRunnerThread run: {e}\n{traceback.format_exc()}")
            self.event_queue.put_nowait(RunnerErrorEvent(self.model_name, f"Unexpected thread error: {e}", []))
        finally:
            if self.loop and self.loop.is_running():
                self.loop.stop()
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            self.is_running = False

    async def run_async(self):
        """
        Asynchronous part of the runner.
        """
        error_occurred = False
        try:
            self.event_queue.put_nowait(RunnerStartedEvent(self.model_name))

            self.runner = LlamaCppRunner(
                model_name=self.model_name,
                model_path=self.model_path,
                llama_cpp_runtime=self.llama_cpp_runtime if self.llama_cpp_runtime is not None else "",
                **self.kwargs
            )
            await self.runner.start()

            if self.runner.port is None:
                raise RuntimeError("Llama.cpp server failed to start or extract port.")

            self.event_queue.put_nowait(PortReadyEvent(self.model_name, self.runner.get_port()))

            self._output_reader_task = self.loop.create_task(self._read_output_continuously())

            if self.runner and self.runner.process:
                await self.runner.process.wait()

            logging.info(f"Llama.cpp process for {self.model_name} exited with code {self.runner.process.returncode if self.runner and self.runner.process else 'N/A'}")

        except Exception as e:
            error_occurred = True
            logging.error(f"Error running LlamaCppRunner: {e}\n{traceback.format_exc()}")
            output_buffer = self.runner.get_output_buffer() if self.runner else []
            self.event_queue.put_nowait(RunnerErrorEvent(self.model_name, str(e), output_buffer))
        finally:
            if self.runner and self.runner.is_running():
                logging.warning(f"Llama.cpp process for {self.model_name} was still running in finally block, stopping.")
                await self.runner.stop()

            if not error_occurred and not self._manual_stop_requested and self.runner and self.runner.process and self.runner.process.returncode != 0:
                error_message = f"Llama.cpp server for {self.model_name} exited with code {self.runner.process.returncode}"
                output_buffer = self.runner.get_output_buffer() if self.runner else []
                logging.error(error_message)
                self.event_queue.put_nowait(RunnerErrorEvent(self.model_name, error_message, output_buffer))

            self.event_queue.put_nowait(RunnerStoppedEvent(self.model_name))

    async def _read_output_continuously(self):
        if not self.runner or not self.runner.process or not self.runner.process.stdout:
            logging.warning("Output stream not available for continuous reading.")
            return

        logging.info(f"Starting continuous output reading for {self.model_name}")
        try:
            while True:
                line = await self.runner.process.stdout.readline()
                if not line:
                    logging.info(f"End of stdout stream reached for {self.model_name}. Stopping continuous reading.")
                    break
                decoded_line = line.decode('utf-8', errors='replace').strip()
                logging.info(f"llama.cpp[{self.model_name}]: {decoded_line}")
        except asyncio.CancelledError:
            logging.info(f"Output reader task for {self.model_name} cancelled.")
        except Exception as e:
            logging.error(f"Error during continuous output reading for {self.model_name}: {e}\n{traceback.format_exc()}")

    def stop(self):
        self.is_running = False
        self._manual_stop_requested = True

        if self._output_reader_task and not self._output_reader_task.done():
            self._output_reader_task.cancel()

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._request_stop_runner(), self.loop)
        else:
            logging.warning(f"Attempted to stop runner {self.model_name} but loop was not running.")

    async def _request_stop_runner(self):
        if self.runner:
            await self.runner.stop()