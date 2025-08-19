#!/usr/bin/env python
import argparse
import signal
import sys
import time
import os

def handle_sigterm(signum, frame):
    """Signal handler for SIGTERM."""
    # In 'ignore_term' mode, we do nothing.
    # In other modes, this allows for graceful shutdown if needed, though we primarily rely on sys.exit().
    if MODE != 'ignore_term':
        sys.exit(0)

def main():
    """
    A dummy llama-server script that simulates different behaviors for testing.
    """
    parser = argparse.ArgumentParser(description="Dummy llama-server for testing.")
    # The script needs to accept a --port argument, as the runner will provide it.
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--alias", type=str, help="Model alias")
    parser.add_argument("--host", type=str, help="Host")


    # Custom argument to control behavior
    parser.add_argument("--mode", choices=['immediate', 'slow', 'ignore_term', 'short_lived'], default='immediate',
                        help="Behavior mode for the dummy server.")

    # Allow unknown arguments to mimic the real llama-server
    args, unknown = parser.parse_known_args()

    global MODE
    MODE = args.mode

    # Register the SIGTERM handler
    signal.signal(signal.SIGTERM, handle_sigterm)

    # Print the startup message that the runner expects
    # The real server prints to stderr, but the runner reads from a merged stream.
    # We hardcode a port here because the dummy server doesn't actually bind to a port.
    print("main: server is listening on http://127.0.0.1:8888", file=sys.stdout, flush=True)

    if MODE == 'immediate':
        sys.exit(0)
    elif MODE == 'short_lived':
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            pass
        sys.exit(0)
    elif MODE == 'slow':
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            pass
        sys.exit(0)
    elif MODE == 'ignore_term':
        # Loop forever, ignoring SIGTERM, until killed by SIGKILL
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                # Also ignore keyboard interrupt
                pass

if __name__ == "__main__":
    main()
