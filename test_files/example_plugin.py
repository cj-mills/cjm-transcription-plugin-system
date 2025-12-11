import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

from cjm_transcription_plugin_system.core import TranscriptionResult, AudioData
from cjm_transcription_plugin_system.plugin_interface import TranscriptionPlugin
from cjm_plugin_system.utils.validation import (
    dict_to_config, SCHEMA_TITLE, SCHEMA_DESC, SCHEMA_ENUM, SCHEMA_MIN, SCHEMA_MAX
)


@dataclass
class WhisperPluginConfig:
    """Configuration for WhisperPlugin."""
    model:str = field(
        default="base",
        metadata={
            SCHEMA_TITLE: "Model",
            SCHEMA_DESC: "Whisper model size. Larger models are more accurate but slower.",
            SCHEMA_ENUM: ["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"]
        }
    )
    device:str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Device",
            SCHEMA_DESC: "Computation device for inference",
            SCHEMA_ENUM: ["cpu", "cuda", "mps", "auto"]
        }
    )
    compute_type:str = field(
        default="default",
        metadata={
            SCHEMA_TITLE: "Compute Type",
            SCHEMA_DESC: "Model precision/quantization",
            SCHEMA_ENUM: ["default", "float16", "float32", "int8", "int8_float16"]
        }
    )
    language:Optional[str] = field(
        default=None,
        metadata={
            SCHEMA_TITLE: "Language",
            SCHEMA_DESC: "Language code (e.g., 'en', 'es', 'fr') or None for auto-detection"
        }
    )
    task:str = field(
        default="transcribe",
        metadata={
            SCHEMA_TITLE: "Task",
            SCHEMA_DESC: "Task to perform",
            SCHEMA_ENUM: ["transcribe", "translate"]
        }
    )
    temperature:float = field(
        default=0.0,
        metadata={
            SCHEMA_TITLE: "Temperature",
            SCHEMA_DESC: "Sampling temperature. 0 for deterministic.",
            SCHEMA_MIN: 0.0,
            SCHEMA_MAX: 1.0
        }
    )
    beam_size:int = field(
        default=5,
        metadata={
            SCHEMA_TITLE: "Beam Size",
            SCHEMA_DESC: "Beam search width.",
            SCHEMA_MIN: 1,
            SCHEMA_MAX: 10
        }
    )
    word_timestamps:bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Word Timestamps",
            SCHEMA_DESC: "Extract word-level timestamps"
        }
    )
    vad_filter:bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "VAD Filter",
            SCHEMA_DESC: "Enable voice activity detection filter"
        }
    )


class WhisperPlugin(TranscriptionPlugin):
    """Example Whisper transcription plugin with dataclass configuration."""

    config_class = WhisperPluginConfig

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config: WhisperPluginConfig = None
        self.model = None
        self.processor = None

    @property
    def name(self) -> str:
        return "whisper"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_formats(self) -> List[str]:
        return ["wav", "mp3", "flac", "m4a", "ogg", "webm"]

    def get_current_config(self) -> WhisperPluginConfig:
        """Return current configuration."""
        return self.config

    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize the Whisper model with configuration."""
        if config is None:
            self.config = WhisperPluginConfig()
        elif isinstance(config, WhisperPluginConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = dict_to_config(WhisperPluginConfig, config, validate=True)
        else:
            raise TypeError(f"Expected WhisperPluginConfig, dict, or None, got {type(config).__name__}")

        self.logger.info(f"Initializing Whisper with config: {self.config}")

        # Mock implementation
        self.model = f"WhisperModel-{self.config.model}"
        self.processor = f"WhisperProcessor-{self.config.device}"

    def execute(self, audio_data: Union[AudioData, str, Path], **kwargs) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        if not self.model:
            raise RuntimeError("Plugin not initialized. Call initialize() first.")

        self.logger.info(f"Transcribing with Whisper model: {self.model}")

        # Mock transcription result
        return TranscriptionResult(
            text=f"Mock transcription using {self.config.model} model",
            confidence=0.95,
            segments=[
                {"start": 0.0, "end": 2.5, "text": "Mock transcription", "confidence": 0.96},
                {"start": 2.5, "end": 5.0, "text": f"using {self.config.model} model", "confidence": 0.94}
            ],
            metadata={
                "model": self.config.model,
                "language": self.config.language or "auto-detected",
                "device": self.config.device,
                "task": self.config.task
            }
        )

    def is_available(self) -> bool:
        """Check if Whisper dependencies are available."""
        return True  # Mock always available

    def cleanup(self) -> None:
        """Clean up model from memory."""
        self.logger.info("Cleaning up Whisper model")
        self.model = None
        self.processor = None
