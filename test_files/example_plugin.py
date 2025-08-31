import logging
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

from cjm_transcription_plugin_system.core import TranscriptionResult, AudioData
from cjm_transcription_plugin_system.plugin_interface import PluginInterface

class WhisperPlugin(PluginInterface):
    """Example Whisper transcription plugin with comprehensive configuration."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config = {}
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
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return comprehensive Whisper configuration schema."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "Whisper Configuration",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                            "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"],
                    "default": "base",
                    "description": "Whisper model size. Larger models are more accurate but slower."
                },
                "device": {
                    "type": "string",
                    "enum": ["cpu", "cuda", "mps", "auto"],
                    "default": "auto",
                    "description": "Computation device for inference"
                },
                "compute_type": {
                    "type": "string",
                    "enum": ["default", "float16", "float32", "int8", "int8_float16"],
                    "default": "default",
                    "description": "Model precision/quantization"
                },
                "language": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": "Language code (e.g., 'en', 'es', 'fr') or null for auto-detection",
                    "examples": ["en", "es", "fr", "de", "ja", "zh", None]
                },
                "task": {
                    "type": "string",
                    "enum": ["transcribe", "translate"],
                    "default": "transcribe",
                    "description": "Task to perform (transcribe keeps original language, translate converts to English)"
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Sampling temperature. 0 for deterministic, higher values for more variation"
                },
                "beam_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Beam search width. Higher values may improve accuracy but are slower"
                },
                "best_of": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Number of candidates when sampling with non-zero temperature"
                },
                "patience": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 1.0,
                    "description": "Beam search patience factor"
                },
                "length_penalty": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Exponential length penalty during beam search"
                },
                "suppress_tokens": {
                    "type": ["array", "string"],
                    "items": {"type": "integer"},
                    "default": "-1",
                    "description": "Token IDs to suppress. '-1' for default suppression, empty array for none"
                },
                "initial_prompt": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": "Optional text to provide as prompt for first window"
                },
                "condition_on_previous_text": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use previous output as prompt for next window"
                },
                "no_speech_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.6,
                    "description": "Threshold for detecting silence"
                },
                "compression_ratio_threshold": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 10.0,
                    "default": 2.4,
                    "description": "Threshold for detecting repetition"
                },
                "logprob_threshold": {
                    "type": "number",
                    "default": -1.0,
                    "description": "Average log probability threshold"
                },
                "word_timestamps": {
                    "type": "boolean",
                    "default": False,
                    "description": "Extract word-level timestamps"
                },
                "prepend_punctuations": {
                    "type": "string",
                    "default": "\"'“¿([{-",
                    "description": "Punctuations to merge with next word"
                },
                "append_punctuations": {
                    "type": "string",
                    "default": "\"'.。,，!！?？:：”)]}、",
                    "description": "Punctuations to merge with previous word"
                },
                "vad_filter": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable voice activity detection filter"
                },
                "vad_parameters": {
                    "type": "object",
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5
                        },
                        "min_speech_duration_ms": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 250
                        },
                        "max_speech_duration_s": {
                            "type": "number",
                            "minimum": 0,
                            "default": 3600
                        }
                    },
                    "default": {}
                }
            },
            "required": ["model"],
            "additionalProperties": False
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Return current configuration with all defaults applied."""
        defaults = self.get_config_defaults()
        current = {**defaults, **self.config}
        
        # Handle nested vad_parameters
        if "vad_parameters" in current and isinstance(current["vad_parameters"], dict):
            vad_defaults = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 3600
            }
            current["vad_parameters"] = {**vad_defaults, **current["vad_parameters"]}
        
        return current
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Whisper model with configuration."""
        if config:
            is_valid, error = self.validate_config(config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {error}")
        
        # Merge with defaults
        defaults = self.get_config_defaults()
        self.config = {**defaults, **(config or {})}
        
        self.logger.info(f"Initializing Whisper with config: {self.config}")
        
        # In a real implementation, this would load the actual Whisper model
        # For example:
        # import whisper
        # self.model = whisper.load_model(self.config["model"], device=self.config["device"])
        
        # Mock implementation
        self.model = f"WhisperModel-{self.config['model']}"
        self.processor = f"WhisperProcessor-{self.config['device']}"
    
    def execute(self, audio_data: Union[AudioData, str, Path], **kwargs) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        if not self.model:
            raise RuntimeError("Plugin not initialized. Call initialize() first.")
        
        # Override config with any provided kwargs
        exec_config = {**self.config, **kwargs}
        
        self.logger.info(f"Transcribing with Whisper model: {self.model}")
        self.logger.info(f"Execution config: {exec_config}")
        
        # In a real implementation, this would:
        # 1. Load/preprocess audio
        # 2. Run Whisper inference
        # 3. Post-process results
        
        # Mock transcription result
        return TranscriptionResult(
            text=f"Mock transcription using {exec_config['model']} model",
            confidence=0.95,
            segments=[
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Mock transcription",
                    "confidence": 0.96
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": f"using {exec_config['model']} model",
                    "confidence": 0.94
                }
            ],
            metadata={
                "model": exec_config["model"],
                "language": exec_config.get("language", "auto-detected"),
                "device": exec_config["device"],
                "task": exec_config["task"]
            }
        )
    
    def is_available(self) -> bool:
        """Check if Whisper dependencies are available."""
        try:
            # In real implementation, check for whisper package
            # import whisper
            # return True
            return True  # Mock always available
        except ImportError:
            return False
    
    def cleanup(self) -> None:
        """Clean up model from memory."""
        self.logger.info("Cleaning up Whisper model")
        self.model = None
        self.processor = None
        # In real implementation: del self.model, torch.cuda.empty_cache(), etc.
