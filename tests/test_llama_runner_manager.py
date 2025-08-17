import asyncio
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_runner.llama_runner_manager import LlamaRunnerManager
from llama_runner.llama_runner_thread import RunnerStoppedEvent, PortReadyEvent

# Mock models and runtimes config
MODELS_CONFIG = {
    "model-1": {"model_path": "/fake/path/model1.gguf"},
    "model-2": {"model_path": "/fake/path/model2.gguf"},
}
RUNTIMES_CONFIG = {"default": {"runtime": "dummy-server"}}

@pytest.fixture
def manager():
    """Fixture to create a LlamaRunnerManager with mock callbacks."""
    def no_op(*args, **kwargs):
        pass

    manager = LlamaRunnerManager(
        models=MODELS_CONFIG,
        llama_runtimes=RUNTIMES_CONFIG,
        default_runtime=RUNTIMES_CONFIG["default"],
        on_started=no_op,
        on_stopped=no_op,
        on_error=no_op,
        on_port_ready=no_op,
    )
    return manager

@pytest.mark.asyncio
@patch('os.path.exists', return_value=True)
@patch('llama_runner.llama_runner_manager.LlamaRunnerThread')
async def test_runner_stop_and_wait_logic(MockLlamaRunnerThread, mock_exists, manager):
    """
    Tests that the manager waits for a running process to stop before starting a new one.
    This test mocks the thread to avoid real threading/event loop issues and focuses on the manager's logic.
    """
    # Start the manager's event processor
    processor = asyncio.create_task(manager._event_processor())

    # --- Setup: Mock a running thread ---
    mock_thread_1 = MagicMock()
    mock_thread_1.is_alive.return_value = True
    manager.llama_runner_threads['model-1'] = mock_thread_1

    # --- Act ---
    # Create a task to request the new runner. This will block until the old one is "stopped".
    request_task = asyncio.create_task(manager.request_runner_start('model-2'))

    # Give the request_task a moment to start and create the stop_future
    await asyncio.sleep(0.1)

    # Check that the stop future exists
    assert 'model-1' in manager._runner_stop_futures

    # Simulate the "stopped" event for the first model, which should unblock the request_task
    await manager.event_queue.put(RunnerStoppedEvent('model-1'))

    # Now, await the completion of the request task
    # It still needs a port to be "ready" to fully return
    # Let's simulate that too.

    # Wait for the startup future for model-2 to be created
    for _ in range(10):
        if 'model-2' in manager._runner_startup_futures:
            break
        await asyncio.sleep(0.1)
    assert 'model-2' in manager._runner_startup_futures

    # Simulate the port ready event for model-2
    await manager.event_queue.put(PortReadyEvent('model-2', 8888))

    # Await the request task
    port2 = await request_task

    # --- Assert ---
    assert port2 == 8888
    mock_thread_1.stop.assert_called_once()
    MockLlamaRunnerThread.assert_called_with(
        model_name='model-2',
        model_path='/fake/path/model2.gguf',
        event_queue=manager.event_queue,
        llama_cpp_runtime='dummy-server',
        **{}
    )

    # --- Cleanup ---
    processor.cancel()
    try:
        await processor
    except asyncio.CancelledError:
        pass
