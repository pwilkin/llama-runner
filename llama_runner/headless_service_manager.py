import logging
import asyncio
from typing import Dict, Any

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

    runner_port_ready_for_proxy = Signal(str, int)
    runner_stopped_for_proxy = Signal(str)

    def on_runner_started_callback(self, model_name: str):
        logger.info(f"Runner started: {model_name}")

    def on_runner_stopped_callback(self, model_name: str):
        logger.info(f"Runner stopped: {model_name}")
        self.runner_stopped_for_proxy.emit(model_name)

    def on_runner_error_callback(self, model_name: str, message: str, output_buffer: list):
        logger.error(f"Runner error for {model_name}: {message}")

    def on_runner_port_ready_callback(self, model_name: str, port: int):
        logger.info(f"Runner port ready for {model_name}: {port}")
        self.runner_port_ready_for_proxy.emit(model_name, port)

    def _initialize_services(self):
        logger.info("Initializing services for headless mode...")
        self.llama_runner_manager = LlamaRunnerManager(
            models=self.models_specific_config,
            llama_runtimes=self.app_config.get('llama-runtimes', {}),
            default_runtime=self.app_config.get('default_runtime', 'llama-server'),
            on_started=lambda name: self._on_runner_event(f"Started {name}"),
            on_stopped=self._on_runner_stopped,
            on_error=self._on_runner_error,
            on_port_ready=self._on_port_ready,
        )
        concurrent_runners = self.app_config.get("concurrentRunners", 1)
        self.llama_runner_manager.set_concurrent_runners_limit(concurrent_runners)
        logger.info("LlamaRunnerManager initialized.")

        proxies_config = self.app_config.get('proxies', {})
        if proxies_config.get('ollama', {}).get('enabled', True):
            logger.info("Ollama Proxy is enabled. Creating server...")
            self.ollama_proxy = OllamaProxyServer(
                all_models_config=self.models_specific_config,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
            )

        if proxies_config.get('lmstudio', {}).get('enabled', True):
            logger.info("LM Studio Proxy is enabled. Creating server...")
            self.lmstudio_proxy = LMStudioProxyServer(
                all_models_config=self.models_specific_config,
                runtimes_config=self.app_config.get('llama-runtimes', {}),
                is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
                on_runner_port_ready=self._on_port_ready,
                on_runner_stopped=self._on_runner_stopped,
            )

        # Connect shutdown signal to stop services
        self.shutdown_signal.connect(self.stop_services)
        logger.info("Headless services initialized.")

    # def _on_model_loaded(self, model_name):
    #     logger.info(f"Model '{model_name}' loaded successfully by LlamaRunnerManager.")

    # def _on_model_load_failed(self, error_message):
    #     logger.error(f"Failed to load model: {error_message}")
    #     # Potentially trigger application shutdown or other error handling
    #     # For now, just log.

    async def stop_services(self):
        """Gracefully stops all managed services."""
        logger.info("Stopping headless services...")

        # The proxy threads stop themselves when their main loop exits
        if self.ollama_proxy and self.ollama_proxy.is_alive():
            logger.info("Stopping Ollama Proxy...")
            self.ollama_proxy.stop()
            self.ollama_proxy.join()
            logger.info("Ollama Proxy stopped.")

        if self.lmstudio_proxy and self.lmstudio_proxy.is_alive():
            logger.info("Stopping LM Studio Proxy...")
            self.lmstudio_proxy.stop()
            self.lmstudio_proxy.join()
            logger.info("LM Studio Proxy stopped.")

        if self.llama_runner_manager:
            logger.info("Stopping LlamaRunnerManager...")
            await self.llama_runner_manager.stop_all_llama_runners_async()
            logger.info("LlamaRunnerManager runners stopped.")

        logger.info("All headless services stopped.")

    def start_services(self):
        """
        This method is implicitly called by __init__ for now.
        If services need to be started separately after instantiation,
        move the start() calls for threads here.
        """
        # LlamaRunnerManager usually loads model on demand or based on config.
        # If an explicit start is needed:
        # self.llama_runner_manager.load_model_async()
        pass