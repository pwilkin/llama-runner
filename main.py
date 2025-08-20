import sys
import logging
import argparse
import os
import signal
import asyncio
import qasync
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from qt_material import apply_stylesheet

from llama_runner.config_loader import CONFIG_DIR, ensure_config_exists, load_config
from llama_runner.main_window import MainWindow
from llama_runner.headless_service_manager import HeadlessServiceManager

def main():
    parser = argparse.ArgumentParser(description="Llama Runner application.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the minimum logging level for console output."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the application in headless mode (no GUI)."
    )
    args = parser.parse_args()

    ensure_config_exists()
    loaded_config = load_config()

    # --- Logging Setup ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
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
        app_file_handler.setLevel(logging.DEBUG)
        app_file_handler.setFormatter(formatter)
        root_logger.addHandler(app_file_handler)
        logging.info(f"App file logging to: {app_log_file_path}")
    except Exception as e:
        logging.error(f"Failed to create app file handler for {app_log_file_path}: {e}")
    # --- End Logging Setup ---

    headless_mode = args.headless

    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        logging.warning("No display environment detected. Forcing headless mode.")
        headless_mode = True

    app = QCoreApplication.instance()
    if app is None:
        if headless_mode:
            app = QCoreApplication(sys.argv)
        else:
            app = QApplication(sys.argv)

    if not headless_mode and isinstance(app, QApplication):
        app.setWindowIcon(QIcon('app_icon.png'))
        apply_stylesheet(app, theme='dark_red.xml', invert_secondary=False)

    # Set up and run the event loop
    asyncio.set_event_loop_policy(qasync.DefaultQEventLoopPolicy())
    loop = asyncio.get_event_loop()

    exit_code = 0
    try:
        if headless_mode:
            hsm = HeadlessServiceManager(loaded_config, loaded_config.get("models", {}))

            async def shutdown_handler():
                logging.info("SIGINT received, shutting down services...")
                await hsm.stop_services()
                app.quit()

            loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown_handler()))

            loop.run_forever()
        else:
            main_window = MainWindow()

            async def shutdown_handler():
                logging.info("Shutdown requested, stopping services...")
                # MainWindow.closeEvent handles the stopping logic
                main_window.close()
                app.quit()

            loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown_handler()))

            main_window.show()
            loop.call_soon(main_window.start_services)
            loop.run_forever()

    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
        exit_code = 1
    finally:
        logging.info("Closing asyncio event loop.")
        loop.close()
        logging.info(f"Application exited with code {exit_code}.")
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
