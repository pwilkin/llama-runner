import subprocess
import requests
import time
import logging
import os
from fastapi import UploadFile
from typing import Optional, Dict, Any, Union


class WhisperServer:
    """
    Manage the startup of the whisper server and interaction with it.
    """

    def __init__(self, audio_config: Dict[str, Any], model_name: str):
        """
        Initialize the server with audio configuration and model name.

        :param audio_config: audio config (normalized)
        :param model_name: model name from audio_config['models']
        """
        self.audio_config = audio_config
        self.model_name = model_name

        # Get model and runtime config
        models = audio_config.get('models', {})
        model_conf = models.get(model_name, {})

        self.runtime_name = model_conf.get('runtime', 'default')
        runtimes = audio_config.get('runtimes', {})
        runtime_conf = runtimes.get(self.runtime_name, {})
        self.runtime_path = runtime_conf.get('runtime')

        if not self.runtime_path:
            raise ValueError(f"Runtime path for '{self.runtime_name}' not defined in audio config.")

        self.model_path = model_conf.get("model_path")
        if not self.model_path:
            raise ValueError(f"Model path for '{self.model_name}' not defined in audio config.")

        # Compose launch command
        self.cmd = [
            self.runtime_path,
            '--model', self.model_path,
        ]

        parameters = model_conf.get("parameters", {})
        if isinstance(parameters, dict):
            for option, value in parameters.items():
                self.cmd.extend([f'--{option}', str(value)])

        # Check if host and port exist, if not add with default values
        default_host = 'localhost'
        default_port = '9000'  # string, since command list elements are strings

        if '--host' not in self.cmd:
            self.cmd.extend(['--host', default_host])
        if '--port' not in self.cmd:
            self.cmd.extend(['--port', default_port])

        # Extract host and port from the command list
        def get_cmd_param(cmd_list, param_name, default):
            try:
                idx = cmd_list.index(param_name)
                return cmd_list[idx + 1]
            except (ValueError, IndexError):
                return default

        self.host = get_cmd_param(self.cmd, '--host', default_host)
        self.port = get_cmd_param(self.cmd, '--port', default_port)
        self.base_url = f'http://{self.host}:{self.port}'

        self.process: Optional[subprocess.Popen] = None

    def start_server(self, wait_seconds: float = 5.0) -> None:
        """Start the whisper server with current parameters."""
        logging.info(f"Starting whisper-server with command: {' '.join(self.cmd)}")
        self.process = subprocess.Popen(self.cmd)
        print(f"Whisper-server started on {self.host}:{self.port} with model {self.model_name}")
        time.sleep(wait_seconds)

    def stop_server(self) -> None:
        """Stop the server if it is running."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("Whisper-server stopped")
        else:
            print("Whisper-server is not running or already stopped.")

    def transcribe_audio(self, audio_path: str) -> Union[Dict[str, Any], None]:
        """Send an audio file to the server for transcription and return the result."""
        url = f"{self.base_url}/inference"
        data = {"response_format": "json"}

        try:
            with open(audio_path, 'rb') as audio_file:
                files = {'file': audio_file}
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            logging.error(f"Error transcribing audio: {e}")
            return None

    def convert_to_wav(self, input_file: UploadFile, output_path: Optional[str] = None) -> str:
        """
        Convert incoming audio file to WAV (16kHz, mono, PCM s16le).

        :param input_file: Uploaded audio file
        :param output_path: Path to save WAV file. Defaults to ~/.llama-runner/temp.wav
        :return: Path to saved WAV file
        """
        if output_path is None:
            output_path = os.path.expanduser("~/.llama-runner/temp.wav")

        input_tmp_dir = os.path.dirname(output_path)
        input_tmp_path = os.path.join(input_tmp_dir, "temp_input")

        os.makedirs(input_tmp_dir, exist_ok=True)

        try:
            with open(input_tmp_path, "wb") as f:
                f.write(input_file.file.read())
        finally:
            input_file.file.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_tmp_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(errors='ignore')
            raise RuntimeError(f"Error during audio conversion: {error_msg}")
        finally:
            if os.path.exists(input_tmp_path):
                os.remove(input_tmp_path)

        return output_path
    
    
    def get_port(self):
        return int(self.port)
