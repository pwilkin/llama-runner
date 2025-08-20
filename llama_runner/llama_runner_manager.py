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

    def get_runner_logs(self, model_name: str) -> List[str]:
        """Get the latest logs from a specific runner.

        Return the runner's output buffer if the runner object exists.
        Previously this required the runner to be 'running' which caused
        the ModelStatusWidget to miss logs that were already present on the
        runner object (e.g. when monitoring started slightly before the
        runner_tasks entry was created). Returning logs whenever the runner
        object exists avoids that race and keeps the UI in sync with the
        Log Viewer.
        """
        runner = self.runners.get(model_name)
        if runner:
            return runner.get_output_buffer()
        return []

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
                logging.info(f"Concurrent runner limit reached. Stopping all existing runners before starting {model_name}.")
                await self.stop_all_llama_runners_async()
            else:
                raise RuntimeError(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached.")
        
        future = asyncio.get_running_loop().create_future()
        self._runner_startup_futures[model_name] = future
    
        logging.info(f"Created future for {model_name}: {id(future)}")
        
        def _on_port_ready_wrapper(name, port):
            fut = self._runner_startup_futures.get(name)
            if fut and not fut.done():
                fut.set_result(port)
            # cleanup
            self._runner_startup_futures.pop(name, None)
            self.on_port_ready(name, port)
    
        def _on_error_wrapper(name, message, output_buffer):
            future = self._runner_startup_futures.get(name)
            if future and not future.done():
                future.set_exception(RuntimeError(message))
            self.on_error(name, message, output_buffer)
    
        model_config = self.models[model_name]
        model_path = model_config.get("model_path")
        llama_cpp_runtime_key = model_config.get("llama_cpp_runtime", "default")
        _raw_llama_cpp_runtime_config = self.llama_runtimes.get(llama_cpp_runtime_key, self.default_runtime)
    
        if isinstance(_raw_llama_cpp_runtime_config, dict):
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config.get("runtime", self.default_runtime)
        else:
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config or self.default_runtime
    
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
        
        # Snapshot current runners and tasks (dicts may mutate)
        # Make sure we only operate on the runners that existed at the start
        runners_to_stop = dict(self.runners)
        tasks_to_stop = dict(self.runner_tasks)
        
        runners = list(runners_to_stop.values())
        tasks = list(tasks_to_stop.values())
        
        # Phase 1: ask all runners to stop (this should let run() exit)
        for runner in runners:
            try:
                await runner.stop()
            except Exception as e:
                # Avoid accessing .model_name on mocks that might not have it
                logging.error(f"Error while stopping runner: {e}")
        
        # Phase 2: give tasks a chance to finish without cancelling first
        pending = [t for t in tasks if not t.done()]
        if pending:
            # Short grace await; no cancellation yet
            await asyncio.gather(*pending, return_exceptions=True)
        
        # Phase 3: if any still pending (rare), cancel and await again
        still_pending = [t for t in tasks if not t.done()]
        if still_pending:
            for t in still_pending:
                t.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
        
        # Phase 4: clear internal state for stopped runners only
        # Remove only the runners that were actually stopped, not new ones
        for model_name in runners_to_stop.keys():
            self.runners.pop(model_name, None)
            self.runner_tasks.pop(model_name, None)