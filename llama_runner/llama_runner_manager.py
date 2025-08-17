import asyncio
import os
import logging
from typing import Optional, Dict, Callable, List

from llama_runner.llama_cpp_runner import LlamaCppRunner


class LlamaRunnerManager:
    def __init__(
        self,
        models: dict,
        llama_runtimes: dict,
        default_runtime: str,
        on_started: Callable[[str], None],
        on_stopped: Callable[[str], None],
        on_error: Callable[[str, str, List[str]], None],
        on_port_ready: Callable[[str, int], None],
    ):
        self.models = models
        self.llama_runtimes = llama_runtimes
        self.default_runtime = default_runtime
        self.on_started = on_started
        self.on_stopped = on_stopped
        self.on_error = on_error
        self.on_port_ready = on_port_ready

        self.runners: Dict[str, LlamaCppRunner] = {}
        self.runner_tasks: Dict[str, asyncio.Task] = {}
        self._runner_startup_futures: Dict[str, asyncio.Future] = {}
        self._runner_stop_futures: Dict[str, asyncio.Future] = {}
        self.concurrent_runners_limit = 1

    def set_concurrent_runners_limit(self, limit: int):
        self.concurrent_runners_limit = limit

    def is_llama_runner_running(self, model_name: str) -> bool:
        return model_name in self.runner_tasks and not self.runner_tasks[model_name].done()

    def get_runner_port(self, model_name: str) -> Optional[int]:
        if self.is_llama_runner_running(model_name):
            runner = self.runners.get(model_name)
            if runner:
                return runner.get_port()
        return None

    async def request_runner_start(self, model_name: str) -> int:
        logging.info(f"Received request to start runner for model: {model_name}")

        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
            logging.info(f"Runner for {model_name} is already starting. Returning existing Future.")
            return await self._runner_startup_futures[model_name]

        if self.is_llama_runner_running(model_name):
            port = self.get_runner_port(model_name)
            if port is not None:
                logging.info(f"Runner for {model_name} is already running on port {port}. Returning port.")
                return port
            else:
                logging.error(f"Runner for {model_name} is reported as running but port is None.")
                raise RuntimeError(f"Runner for {model_name} is running but port is unavailable.")

        running_runners = {name: task for name, task in self.runner_tasks.items() if not task.done()}
        if len(running_runners) >= self.concurrent_runners_limit:
            if self.concurrent_runners_limit == 1:
                models_to_stop = list(running_runners.keys())
                logging.info(f"Concurrent runner limit reached. Stopping existing runner(s): {models_to_stop} before starting {model_name}.")
                for name_to_stop in models_to_stop:
                    await self.stop_llama_runner(name_to_stop)
            else:
                raise RuntimeError(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached.")

        future = asyncio.get_running_loop().create_future()
        self._runner_startup_futures[model_name] = future

        def _on_port_ready_wrapper(name, port):
            if name in self._runner_startup_futures:
                self._runner_startup_futures[name].set_result(port)
            self.on_port_ready(name, port)

        def _on_error_wrapper(name, message, output_buffer):
            if name in self._runner_startup_futures:
                self._runner_startup_futures[name].set_exception(RuntimeError(message))
            self.on_error(name, message, output_buffer)

        model_config = self.models[model_name]
        model_path = model_config.get("model_path")
        llama_cpp_runtime_key = model_config.get("llama_cpp_runtime", "default")
        _raw_llama_cpp_runtime_config = self.llama_runtimes.get(llama_cpp_runtime_key, self.default_runtime)

        if isinstance(_raw_llama_cpp_runtime_config, dict):
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config.get("runtime")
        else:
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config

        if not all([llama_cpp_runtime_command, model_path, os.path.exists(model_path)]):
             raise RuntimeError("Invalid configuration or file not found.")

        runner = LlamaCppRunner(
            model_name=model_name,
            model_path=model_path,
            llama_cpp_runtime=llama_cpp_runtime_command,
            on_started=self.on_started,
            on_stopped=self.on_stopped,
            on_error=_on_error_wrapper,
            on_port_ready=_on_port_ready_wrapper,
            **model_config.get("parameters", {})
        )
        self.runners[model_name] = runner
        task = asyncio.create_task(runner.run())
        self.runner_tasks[model_name] = task
        return await future

    async def stop_llama_runner(self, model_name: str):
        logging.info(f"Stopping Llama Runner for {model_name}...")
        if model_name in self.runner_tasks:
            task = self.runner_tasks[model_name]
            runner = self.runners[model_name]

            if not task.done():
                await runner.stop()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logging.info(f"Task for {model_name} cancelled successfully.")

            del self.runner_tasks[model_name]
            del self.runners[model_name]
        else:
            logging.warning(f"Attempted to stop a non-existent runner: {model_name}")

    async def stop_all_llama_runners_async(self):
        logging.info("Stopping all Llama Runners asynchronously...")
        running_tasks = list(self.runner_tasks.keys())
        for model_name in running_tasks:
            await self.stop_llama_runner(model_name)