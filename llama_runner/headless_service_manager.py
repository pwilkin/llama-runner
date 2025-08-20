import logging
import asyncio
from typing import List, Dict
from collections import defaultdict

from llama_runner.llama_runner_manager import LlamaRunnerManager
from llama_runner.ollama_proxy_thread import OllamaProxyServer
from llama_runner.lmstudio_proxy_thread import LMStudioProxyServer
from llama_runner.log_parser import LlamaLogParser, ModelStatusInfo

logger = logging.getLogger(__name__)

class HeadlessServiceManager:
    def __init__(self, app_config, model_config):
        self.app_config = app_config
        self.models_specific_config = model_config
        self.llama_runner_manager: LlamaRunnerManager | None = None
        self.ollama_proxy: OllamaProxyServer | None = None
        self.lmstudio_proxy: LMStudioProxyServer | None = None
        self.log_monitors: Dict[str, asyncio.Task] = {}
        self.log_parsers: Dict[str, LlamaLogParser] = {}
        self.previous_status: Dict[str, ModelStatusInfo] = {}
        self._initialize_services()

    def _on_runner_error(self, model_name: str, message: str, output_buffer: List[str]):
        logger.error(f"Runner error for {model_name}: {message}")
        # Stop log monitoring for this model
        if model_name in self.log_monitors:
            self.log_monitors[model_name].cancel()
            del self.log_monitors[model_name]

    def _on_runner_event(self, message: str):
        logger.info(f"Runner Manager Event: {message}")

    def _on_runner_started(self, model_name: str):
        self._on_runner_event(f"Started {model_name}")
        # Start log monitoring for this model
        if model_name not in self.log_monitors:
            self.log_parsers[model_name] = LlamaLogParser()
            self.previous_status[model_name] = ModelStatusInfo(status=self.log_parsers[model_name].parse_multiple_lines([]).status)
            self.log_monitors[model_name] = asyncio.create_task(self._monitor_model_logs(model_name))

    def _on_runner_stopped(self, model_name: str):
        self._on_runner_event(f"Stopped {model_name}")
        # Stop log monitoring for this model
        if model_name in self.log_monitors:
            self.log_monitors[model_name].cancel()
            del self.log_monitors[model_name]
            if model_name in self.previous_status:
                del self.previous_status[model_name]
            if model_name in self.log_parsers:
                del self.log_parsers[model_name]

    def _initialize_services(self):
        logger.info("Initializing services for headless mode...")
        self.llama_runner_manager = LlamaRunnerManager(
            models=self.models_specific_config,
            llama_runtimes=self.app_config.get("llama-runtimes", {}),
            default_runtime=self.app_config.get("default_runtime", "llama-server"),
            on_started=self._on_runner_started,
            on_stopped=self._on_runner_stopped,
            on_error=self._on_runner_error,
            on_port_ready=lambda name, port: self._on_runner_event(f"Port {port} ready for {name}"),
        )
        self.llama_runner_manager.set_concurrent_runners_limit(
            self.app_config.get("concurrentRunners", 1)
        )
        logger.info("LlamaRunnerManager initialized.")

        proxies_config = self.app_config.get("proxies", {})
        if proxies_config.get("ollama", {}).get("enabled", True):
            logger.info("Ollama Proxy is enabled. Creating server...")
            self.ollama_proxy = OllamaProxyServer(
                all_models_config=self.models_specific_config,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
            )

        if proxies_config.get("lmstudio", {}).get("enabled", True):
            logger.info("LM Studio Proxy is enabled. Creating server...")
            self.lmstudio_proxy = LMStudioProxyServer(
                all_models_config=self.models_specific_config,
                runtimes_config=self.app_config.get("llama-runtimes", {}),
                is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
                on_runner_port_ready=lambda name, port: None,
                on_runner_stopped=lambda name: None,
            )

    async def _monitor_model_logs(self, model_name: str):
        """Monitor logs for a specific model and output status to stdout."""
        while True:
            try:
                if self.llama_runner_manager and self.llama_runner_manager.is_llama_runner_running(model_name):
                    logs = self.llama_runner_manager.get_runner_logs(model_name)
                    if logs and model_name in self.log_parsers:
                        parser = self.log_parsers[model_name]
                        status_info = parser.parse_multiple_lines(logs)
                        
                        # Only output if status has changed
                        if model_name not in self.previous_status or self.previous_status[model_name] != status_info:
                            status_text = parser.format_status_text(status_info)
                            print(f"[{model_name}] Status: {status_text}")
                            self.previous_status[model_name] = status_info
                else:
                    # Model is not running, stop monitoring
                    break
                
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring logs for {model_name}: {e}")
                break

    def update_config(self, new_config):
        """
        Updates the configuration and reinitializes services as needed.
        """
        logger.info("Updating configuration in HeadlessServiceManager.")
        self.app_config = new_config
        self._initialize_services()

    async def stop_services(self):
        logger.info("Stopping headless services...")

        # Cancel all log monitoring tasks
        for task in self.log_monitors.values():
            task.cancel()

        tasks_to_cancel = []
        if self.ollama_proxy and self.ollama_proxy.task:
            tasks_to_cancel.append(self.ollama_proxy.task)
            self.ollama_proxy.stop()
        if self.lmstudio_proxy and self.lmstudio_proxy.task:
            tasks_to_cancel.append(self.lmstudio_proxy.task)
            self.lmstudio_proxy.stop()

        if self.llama_runner_manager:
            await self.llama_runner_manager.stop_all_llama_runners_async()

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Wait for log monitoring tasks to complete
        if self.log_monitors:
            await asyncio.gather(*self.log_monitors.values(), return_exceptions=True)

        logger.info("All headless services stopped.")

