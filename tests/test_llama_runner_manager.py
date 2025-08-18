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

    return LlamaRunnerManager(
        models=MODELS_CONFIG,
        llama_runtimes=RUNTIMES_CONFIG,
        default_runtime=RUNTIMES_CONFIG["default"],
        on_started=no_op,
        on_stopped=no_op,
        on_error=no_op,
        on_port_ready=no_op,
    )


@pytest.mark.asyncio
@patch("os.path.exists", return_value=True)
@patch("llama_runner.llama_runner_manager.LlamaCppRunner")
async def test_runner_stop_and_wait_logic(MockLlamaCppRunner, mock_exists, manager):
    """
    Tests that the manager waits for a running process to stop before starting a new one.
    This test mocks the LlamaCppRunner to focus on the manager's logic.
    """

    # Prepare two runner mocks with AsyncMock run/stop and stop events
    stop_event_1 = asyncio.Event()

    async def fake_run_1():
        await stop_event_1.wait()
        return  # exit normally when stop_event is set

    async def fake_stop_1():
        stop_event_1.set()

    mock_runner_1 = MagicMock()
    mock_runner_1.run = fake_run_1
    mock_runner_1.stop = fake_stop_1

    stop_event_2 = asyncio.Event()

    async def fake_run_2():
        await stop_event_2.wait()
        return

    async def fake_stop_2():
        stop_event_2.set()

    mock_runner_2 = MagicMock()
    mock_runner_2.run = fake_run_2
    mock_runner_2.stop = fake_stop_2

    # --- Side effect factory for constructor ---
    runners = [mock_runner_1, mock_runner_2]
    call_index = {"i": 0}

    def ctor_side_effect(*args, **kwargs):
        mock_obj = runners[call_index["i"]]
        call_index["i"] += 1
        # Attach constructor kwargs (like on_port_ready wrapper) to the mock
        for k, v in kwargs.items():
            setattr(mock_obj, k, v)
        return mock_obj

    MockLlamaCppRunner.side_effect = ctor_side_effect

    # --- Act: Start the first runner ---
    start_task_1 = asyncio.create_task(manager.request_runner_start("model-1"))
    await asyncio.sleep(0)  # let ctor run and wrapper attach

    # Trigger the wrapper stored on the mock runner
    mock_runner_1.on_port_ready("model-1", 8888)
    port1 = await asyncio.wait_for(start_task_1, timeout=1.0)

    assert port1 == 8888
    assert "model-1" in manager.runners
    assert "model-1" in manager.runner_tasks

    # --- Act: Start the second runner, which should stop the first ---
    start_task_2 = asyncio.create_task(manager.request_runner_start("model-2"))
    await asyncio.sleep(0)

    mock_runner_2.on_port_ready("model-2", 9999)
    port2 = await asyncio.wait_for(start_task_2, timeout=1.0)

    # --- Assert ---
    assert port2 == 9999
    mock_runner_1.stop.assert_called_once()
    assert "model-1" not in manager.runners
    assert "model-1" not in manager.runner_tasks
    assert "model-2" in manager.runners
    assert "model-2" in manager.runner_tasks

    # --- Cleanup ---
    await asyncio.wait_for(manager.stop_all_llama_runners_async(), timeout=1.0)
    assert not manager.runners
    assert not manager.runner_tasks
