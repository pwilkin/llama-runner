import logging
import asyncio
from typing import Dict, Any, List, Optional

from llama_runner.llama_runner_manager import LlamaRunnerManager
from llama_runner.ollama_proxy_thread import OllamaProxyServer
from llama_runner.lmstudio_proxy_thread import LMStudioProxyServer

logger = logging.getLogger(__name__)

class HeadlessServiceManager:
    def __init__(self, app_config, model_config):
        self.app_config = app_config
        self.models_specific_config = model_config
        self.llama_runner_manager: LlamaRunnerManager | None = None
        self.ollama_proxy: OllamaProxyServer | None = None
        self.lmstudio_proxy: LMStudioProxyServer | None = None
        self._initialize_services()

    def _on_runner_error(self, model_name: str, message: str, output_buffer: List[str]):
        logger.error(f"Runner error for {model_name}: {message}")

    def _on_runner_event(self, message: str):
        logger.info(f"Runner Manager Event: {message}")

    def _initialize_services(self):
        logger.info("Initializing services for headless mode...")
        self.llama_runner_manager = LlamaRunnerManager(
            models=self.models_specific_config,
            llama_runtimes=self.app_config.get("llama-runtimes", {}),
            default_runtime=self.app_config.get("default_runtime", "llama-server"),
            on_started=lambda name: self._on_runner_event(f"Started {name}"),
            on_stopped=lambda name: self._on_runner_event(f"Stopped {name}"),
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

    async def stop_services(self):
        logger.info("Stopping headless services...")

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

        logger.info("All headless services stopped.")

