import sys
import logging # Import logging
import argparse # Import argparse
import os # Import os
import signal # Import signal for SIGINT handling
from datetime import datetime # Import datetime
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QTimer # Added QCoreApplication, QTimer
from PySide6.QtGui import QIcon # Import QIcon

# Import CONFIG_DIR and ensure_config_exists, load_config
from llama_runner.config_loader import CONFIG_DIR, ensure_config_exists, load_config

# Import the MainWindow class from your UI file
# Import this *after* configuring logging
from llama_runner.main_window import MainWindow
# Import HeadlessServiceManager
from llama_runner.headless_service_manager import HeadlessServiceManager


def main():
    """
    Initializes and runs the PySide6 application.
    Configures logging to console and a file.
    Supports GUI and headless modes.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Llama Runner application.")
    parser.add_argument(
        "--log-level",
        default="INFO", # Default to INFO
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the minimum logging level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    # --log-prompts argument is removed, will be controlled by config.json
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the application in headless mode (no GUI)."
    )
    args = parser.parse_args()

    # Ensure config directory exists for log files and load config early
    ensure_config_exists()
    loaded_config = load_config() # Load config once at the start

    # --- Logging Setup ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Process all messages, handlers filter

    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    app_log_file_path = os.path.join(CONFIG_DIR, "app.log")
    try:
        app_file_handler = logging.FileHandler(app_log_file_path)
        app_file_handler.setLevel(logging.DEBUG) # Capture all to app.log
        app_file_handler.setFormatter(formatter)
        root_logger.addHandler(app_file_handler)
        logging.info(f"App file logging to: {app_log_file_path}")
    except Exception as e:
        logging.error(f"Failed to create app file handler for {app_log_file_path}: {e}")

    prompts_logger = logging.getLogger("prompts")
    prompts_logger.setLevel(logging.DEBUG)
    
    # Determine prompt logging from config
    prompt_logging_enabled_from_config = loaded_config.get('logging', {}).get('prompt_logging_enabled', False)

    if prompt_logging_enabled_from_config:
        prompt_log_filename = f"prompts-{datetime.now().strftime('%Y%m%d')}.log"
        prompt_log_file_path = os.path.join(CONFIG_DIR, prompt_log_filename)
        try:
            prompt_file_handler = logging.FileHandler(prompt_log_file_path)
            prompt_file_handler.setFormatter(formatter)
            prompt_file_handler.setLevel(logging.INFO)
            prompts_logger.addHandler(prompt_file_handler)
            logging.info(f"Prompt logging enabled (from config) to: {prompt_log_file_path}")
        except Exception as e:
            logging.error(f"Failed to create prompt file handler for {prompt_log_file_path}: {e}")
    else:
        logging.info("Prompt logging disabled (from config).")
    # --- End Logging Setup ---

    logging.info(f"Console logging level: {args.log_level.upper()}")
    logging.info(f"Prompt logging (from config): {'enabled' if prompt_logging_enabled_from_config else 'disabled'}")

    headless_mode = args.headless
    app_instance = None
    main_component = None # Will be MainWindow or HeadlessServiceManager

    if not headless_mode:
        try:
            # Attempt to initialize QApplication for GUI mode
            # Check if XDG_RUNTIME_DIR is set, common requirement for Wayland/X11 GUI apps
            if sys.platform.startswith('linux') and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                logging.warning("DISPLAY or WAYLAND_DISPLAY environment variable not set. GUI mode might fail.")
                logging.warning("Attempting to run in headless mode due to missing display environment.")
                headless_mode = True # Force headless if no display
            
            if not headless_mode: # Re-check in case it was forced above
                app_instance = QApplication.instance() # Check if already exists
                if not app_instance:
                    app_instance = QApplication(sys.argv)
                
                # setWindowIcon should only be called on QApplication instance
                if isinstance(app_instance, QApplication):
                    app_instance.setWindowIcon(QIcon('app_icon.png'))
                
                # MainWindow now loads its config internally, including prompt logging status
                main_component = MainWindow() # No longer pass prompt_logging_enabled
                main_component.show()
                logging.info("Application started in GUI mode.")

        except Exception as e:
            logging.error(f"Failed to initialize QApplication (GUI mode): {e}. Switching to headless mode.")
            headless_mode = True
            if app_instance: # If QApplication was partially created, try to clean up
                app_instance.quit() # Request quit
                app_instance = None


    if headless_mode:
        logging.info("Application starting in headless mode.")
        app_instance = QCoreApplication.instance() # Check if already exists
        if not app_instance:
            app_instance = QCoreApplication(sys.argv)

        # loaded_config is already available from the top of main()
        if not loaded_config: # Check if the main config dictionary is empty (error case)
            logging.critical("Failed to load configuration from config.json. Cannot start headless mode.")
            sys.exit(1)

        # HeadlessServiceManager expects the full config and then the 'models' part
        hsm_app_config = loaded_config
        hsm_model_config = loaded_config.get("models", {}) # Get the 'models' dictionary

        main_component = HeadlessServiceManager(hsm_app_config, hsm_model_config)
        
        # Connect the HeadlessServiceManager's shutdown signal to QCoreApplication's quit
        # Ensure app_instance is valid before connecting signals
        if app_instance:
            main_component.shutdown_signal.connect(app_instance.quit)
            # Also connect QCoreApplication's aboutToQuit to the service manager's stop
            app_instance.aboutToQuit.connect(main_component.stop_services)
        else:
            logging.error("app_instance is None, cannot connect signals for HeadlessServiceManager.")
            sys.exit(1) # Critical error if app_instance is not set in headless mode

        # SIGINT Handler for graceful shutdown in headless mode
        # This needs to be defined where main_component is in scope (it's the HeadlessServiceManager instance)
        # and QCoreApplication is available.
        def sigint_handler(signum, frame):
            logging.info("SIGINT received, requesting application quit...")
            app = QCoreApplication.instance()
            if app:
                app.quit() # Just tell Qt to quit
            else:
                logging.warning("QCoreApplication.instance() is None in SIGINT handler. Cannot quit.")

        signal.signal(signal.SIGINT, sigint_handler)
        logging.info("Registered SIGINT handler for graceful shutdown in headless mode.")

        logging.info("HeadlessServiceManager initialized.")
        # In headless mode, services are started by HeadlessServiceManager's __init__

        # Timer to allow Python to process signals
        # QTimer is already imported from PySide6.QtCore
        signal_processing_timer = QTimer()
        signal_processing_timer.timeout.connect(lambda: None)
        signal_processing_timer.start(200) # ms
        logging.info("Started QTimer for signal processing in headless mode.")

    if not app_instance:
        logging.critical("Failed to initialize application (QApplication or QCoreApplication). Exiting.")
        sys.exit(1)

    # Start the Qt event loop
    exit_code = app_instance.exec()
    logging.info(f"Application exited with code {exit_code}.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
