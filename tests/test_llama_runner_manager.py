import asyncio
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_runner.llama_runner_manager import LlamaRunnerManager

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
        default_runtime=RUNTIMES_CONFIG["default"]["runtime"],
        on_started=no_op,
        on_stopped=no_op,
        on_error=no_op,
        on_port_ready=no_op,
    )
    return manager

@pytest.mark.asyncio
@patch('os.path.exists', return_value=True)
@patch('llama_runner.llama_runner_manager.LlamaCppRunner')
async def test_runner_stop_and_wait_logic(MockLlamaCppRunner, mock_exists, manager):
    """
    Tests that the manager waits for a running process to stop before starting a new one.
    This test mocks the LlamaCppRunner to focus on the manager's logic.
    """
    # --- Setup: Mock LlamaCppRunner instances ---
    mock_runner_1 = MagicMock()
    mock_runner_1.run = MagicMock(return_value=asyncio.sleep(10)) # Simulate a long running process
    mock_runner_1.stop = MagicMock(return_value=asyncio.sleep(0)) # stop is a coroutine

    mock_runner_2 = MagicMock()
    mock_runner_2.run = MagicMock(return_value=asyncio.sleep(10))

    # Configure the mock to return different instances for model-1 and model-2
    MockLlamaCppRunner.side_effect = [mock_runner_1, mock_runner_2]

    # --- Act: Start the first runner ---
    start_task_1 = asyncio.create_task(manager.request_runner_start('model-1'))
    await asyncio.sleep(0.1) # allow task to start

    # Manually trigger the port ready callback to resolve the future
    manager.on_port_ready('model-1', 8888)
    port1 = await start_task_1

    assert port1 == 8888
    assert 'model-1' in manager.runners
    assert 'model-1' in manager.runner_tasks

    # --- Act: Start the second runner, which should stop the first ---
    start_task_2 = asyncio.create_task(manager.request_runner_start('model-2'))
    await asyncio.sleep(0.1) # allow task to start

    # Manually trigger the port ready callback for the second model
    manager.on_port_ready('model-2', 9999)
    port2 = await start_task_2

    # --- Assert ---
    assert port2 == 9999

    # Check that the first runner was stopped
    mock_runner_1.stop.assert_called_once()

    # Check that the first runner's task is no longer in the manager
    assert 'model-1' not in manager.runners
    assert 'model-1' not in manager.runner_tasks

    # Check that the second runner is now managed
    assert 'model-2' in manager.runners
    assert 'model-2' in manager.runner_tasks

    # --- Cleanup ---
    # Stop the remaining runner
    await manager.stop_all_llama_runners_async()
    assert not manager.runners
    assert not manager.runner_tasks
