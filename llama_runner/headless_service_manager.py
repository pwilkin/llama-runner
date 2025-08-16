import logging
import asyncio
from PySide6.QtCore import QObject, Signal

from llama_runner.llama_runner_manager import LlamaRunnerManager
from llama_runner.ollama_proxy_thread import OllamaProxyThread
from llama_runner.lmstudio_proxy_thread import FastAPIProxyThread as LMStudioProxyThread # Corrected import

logger = logging.getLogger(__name__)

class HeadlessServiceManager(QObject):
    """
    Manages the core services of the application in headless mode.
    Initializes and controls LlamaRunnerManager and API proxy threads.
    """
    shutdown_signal = Signal() # Changed from pyqtSignal

    def __init__(self, app_config, model_config, parent=None): # model_config here is app_config['models']
        super().__init__(parent)
        self.app_config = app_config
        # model_config passed in is actually app_config['models']
        # For clarity, let's use a more descriptive name internally if needed,
        # or just ensure it's used correctly.
        self.models_specific_config = model_config

        self.llama_runner_manager = None
        self.ollama_proxy = None
        self.lmstudio_proxy = None

        self._initialize_services()

    def _on_runner_event(self, message: str):
        # In headless mode, we just log the events.
        logger.info(f"Runner Manager Event: {message}")

    def _on_runner_error(self, model_name: str, message: str, output_buffer: list):
        logger.error(f"Runner error for {model_name}: {message}")

    def _initialize_services(self):
        """Initializes all managed services."""
        logger.info("Initializing services for headless mode...")

        # Initialize LlamaRunnerManager
        self.llama_runner_manager = LlamaRunnerManager(
            models=self.models_specific_config,
            llama_runtimes=self.app_config.get('llama-runtimes', {}),
            default_runtime=self.app_config.get('default_runtime', 'llama-server'),
            on_started=lambda name: self._on_runner_event(f"Started {name}"),
            on_stopped=lambda name: self._on_runner_event(f"Stopped {name}"),
            on_error=self._on_runner_error,
            on_port_ready=lambda name, port: self._on_runner_event(f"Port {port} ready for {name}"),
        )
        # Start the manager's event processor
        asyncio.create_task(self.llama_runner_manager._event_processor())

        concurrent_runners = self.app_config.get("concurrentRunners", 1)
        if not isinstance(concurrent_runners, int) or concurrent_runners < 1:
            logger.warning(f"Invalid 'concurrentRunners' value: {concurrent_runners}. Defaulting to 1.")
            concurrent_runners = 1
        self.llama_runner_manager.set_concurrent_runners_limit(concurrent_runners)
        logger.info("LlamaRunnerManager initialized.")

        # Get proxy and logging settings from the unified config
        proxies_config = self.app_config.get('proxies', {})
        ollama_proxy_settings = proxies_config.get('ollama', {})
        lmstudio_proxy_settings = proxies_config.get('lmstudio', {})
        logging_settings = self.app_config.get('logging', {})

        ollama_enabled = ollama_proxy_settings.get('enabled', True)
        lmstudio_enabled = lmstudio_proxy_settings.get('enabled', True)
        lmstudio_api_key = lmstudio_proxy_settings.get('api_key', None)
        prompt_logging_enabled = logging_settings.get('prompt_logging_enabled', False)

        # Initialize Ollama Proxy if enabled
        if ollama_enabled:
            logger.info("Ollama Proxy is enabled. Initializing...")
            self.ollama_proxy = OllamaProxyThread(
                all_models_config=self.models_specific_config,
                runtimes_config=self.app_config.get('llama-runtimes', {}),
                is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
                prompt_logging_enabled=prompt_logging_enabled,
                prompts_logger=logging.getLogger("prompts")
            )
            self.ollama_proxy.start()
            logger.info("Ollama Proxy thread started.")
        else:
            logger.info("Ollama Proxy is disabled in config.")

        # Initialize LM Studio Proxy if enabled
        if lmstudio_enabled:
            logger.info("LM Studio Proxy is enabled. Initializing...")
            self.lmstudio_proxy = LMStudioProxyThread( # Using the aliased FastAPIProxyThread
                all_models_config=self.models_specific_config,
                runtimes_config=self.app_config.get('llama-runtimes', {}),
                is_model_running_callback=self.llama_runner_manager.is_llama_runner_running,
                get_runner_port_callback=self.llama_runner_manager.get_runner_port,
                request_runner_start_callback=self.llama_runner_manager.request_runner_start,
                prompt_logging_enabled=prompt_logging_enabled,
                prompts_logger=logging.getLogger("prompts"),
                api_key=lmstudio_api_key
            )
            self.lmstudio_proxy.start()
            logger.info("LM Studio Proxy thread started.")
        else:
            logger.info("LM Studio Proxy is disabled in config.")

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