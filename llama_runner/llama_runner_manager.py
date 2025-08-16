import asyncio
import os
import logging
from typing import Optional, Dict, Any, Union, Callable, List

from llama_runner.llama_runner_thread import (
    LlamaRunnerThread,
    RunnerStartedEvent,
    RunnerStoppedEvent,
    RunnerErrorEvent,
    PortReadyEvent,
)


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

        self.event_queue = asyncio.Queue()
        self.llama_runner_threads: Dict[str, LlamaRunnerThread] = {}
        self._runner_startup_futures: Dict[str, asyncio.Future] = {}
        self._runner_stop_futures: Dict[str, asyncio.Future] = {}
        self._current_running_model: Optional[str] = None
        self.concurrent_runners_limit = 1

    def set_concurrent_runners_limit(self, limit: int):
        self.concurrent_runners_limit = limit

    def is_llama_runner_running(self, model_name: str) -> bool:
        thread = self.llama_runner_threads.get(model_name)
        return bool(thread and thread.is_alive() and thread.is_running)

    def get_runner_port(self, model_name: str) -> Optional[int]:
        thread = self.llama_runner_threads.get(model_name)
        if thread and thread.is_alive() and thread.runner and thread.runner.is_running():
            return thread.runner.get_port()
        return None

    async def _event_processor(self):
        logging.info("Event processor started.")
        while True:
            try:
                event = await self.event_queue.get()
                if isinstance(event, RunnerStartedEvent):
                    self.on_started(event.model_name)
                elif isinstance(event, RunnerStoppedEvent):
                    if event.model_name in self.llama_runner_threads:
                        del self.llama_runner_threads[event.model_name]
                    self.on_stopped(event.model_name)
                    if event.model_name in self._runner_stop_futures:
                        self._runner_stop_futures.pop(event.model_name).set_result(True)
                    if event.model_name in self._runner_startup_futures:
                        self._runner_startup_futures.pop(event.model_name).set_exception(RuntimeError(f"Runner for {event.model_name} stopped unexpectedly during startup."))
                elif isinstance(event, RunnerErrorEvent):
                    self.on_error(event.model_name, event.message, event.output_buffer)
                    if event.model_name in self._runner_startup_futures:
                        self._runner_startup_futures.pop(event.model_name).set_exception(RuntimeError(event.message))
                    if event.model_name in self._runner_stop_futures:
                        self._runner_stop_futures.pop(event.model_name).set_exception(RuntimeError(event.message))
                elif isinstance(event, PortReadyEvent):
                    self.on_port_ready(event.model_name, event.port)
                    if event.model_name in self._runner_startup_futures:
                        self._runner_startup_futures.pop(event.model_name).set_result(event.port)
                self.event_queue.task_done()
            except Exception as e:
                logging.error(f"Error in event processor: {e}", exc_info=True)

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

        running_runners = {name: thread for name, thread in self.llama_runner_threads.items() if thread.is_alive()}
        if len(running_runners) >= self.concurrent_runners_limit:
            if self.concurrent_runners_limit == 1:
                models_to_stop = list(running_runners.keys())
                logging.info(f"Concurrent runner limit reached. Stopping existing runner(s): {models_to_stop} before starting {model_name}.")
                for name_to_stop in models_to_stop:
                    stop_future = asyncio.get_running_loop().create_future()
                    self._runner_stop_futures[name_to_stop] = stop_future
                    self.stop_llama_runner(name_to_stop)
                    try:
                        logging.info(f"Waiting for runner {name_to_stop} to stop...")
                        await asyncio.wait_for(stop_future, timeout=30.0)
                        logging.info(f"Runner {name_to_stop} stopped successfully.")
                    except asyncio.TimeoutError:
                        logging.error(f"Timeout waiting for runner {name_to_stop} to stop.")
                        if name_to_stop in self._runner_stop_futures:
                            del self._runner_stop_futures[name_to_stop]
                        raise RuntimeError(f"Timeout waiting for runner {name_to_stop} to stop.")
            else:
                raise RuntimeError(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached.")

        future = asyncio.get_running_loop().create_future()
        self._runner_startup_futures[model_name] = future

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

        thread = LlamaRunnerThread(
            model_name=model_name,
            model_path=model_path,
            event_queue=self.event_queue,
            llama_cpp_runtime=llama_cpp_runtime_command,
            **model_config.get("parameters", {})
        )
        self.llama_runner_threads[model_name] = thread
        thread.start()
        return await future

    def stop_llama_runner(self, model_name: str):
        if model_name in self.llama_runner_threads:
            print(f"Stopping Llama Runner for {model_name}...")
            thread = self.llama_runner_threads[model_name]
            if thread.is_alive():
                thread.stop()
            else:
                logging.warning(f"Attempted to stop non-running thread {model_name}. Cleaning up.")
                if model_name in self._runner_stop_futures:
                    self._runner_stop_futures.pop(model_name).set_result(True)
                self.on_stopped(model_name)
        else:
            logging.warning(f"Attempted to stop a non-existent runner: {model_name}")

    async def stop_all_llama_runners_async(self):
        print("Stopping all Llama Runners asynchronously...")
        running_threads = [name for name, thread in self.llama_runner_threads.items() if thread.is_alive()]
        if not running_threads:
            return
        stop_futures = []
        for model_name in running_threads:
            if model_name not in self._runner_stop_futures:
                stop_future = asyncio.get_running_loop().create_future()
                self._runner_stop_futures[model_name] = stop_future
                stop_futures.append(stop_future)
            else:
                stop_futures.append(self._runner_stop_futures[model_name])
            self.stop_llama_runner(model_name)
        if stop_futures:
            await asyncio.gather(*stop_futures, return_exceptions=True)