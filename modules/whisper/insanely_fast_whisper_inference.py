import threading
import os
import time
import numpy as np
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import torch
from transformers import pipeline, Pipeline
from transformers.utils import is_flash_attn_2_available
import gradio as gr
from huggingface_hub import hf_hub_download
import whisper
import gc
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from argparse import Namespace

from modules.utils.paths import INSANELY_FAST_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR
from modules.whisper.data_classes import WhisperParams, Segment
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class InsanelyFastWhisperInference(BaseTranscriptionPipeline):
    """
    High-performance Whisper inference pipeline using HuggingFace transformers.

    Manages loading, caching and offloading of models, and performs audio transcription
    (and optional translation) with progress reporting.
    """
    _model: Optional[Pipeline] = None
    _model_lock = threading.RLock()
    _current_model_size: Optional[str] = None
    _current_compute_type: Optional[str] = None
    _offload_timer: Optional[threading.Timer] = None

    def __init__(
        self,
        model_dir: str = INSANELY_FAST_WHISPER_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        """
        Initialize the inference engine.

        Args:
            model_dir (str): Path where Whisper model directories are stored.
            diarization_model_dir (str): Path to speaker-diarization models.
            uvr_model_dir (str): Path to music-separation (UVR) models.
            output_dir (str): Path to write output artifacts (optional).
        """
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.available_models = self.get_model_paths()

    def transcribe(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        progress: Optional[gr.Progress] = None,
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        """
        Perform transcription on an audio input.

        Args:
            audio (str | np.ndarray | torch.Tensor):
                File path, raw array or tensor of audio samples.
            progress (gr.Progress, optional):
                Gradio progress callback to update UI.
            progress_callback (Callable, optional):
                Low-level callback invoked with float progress values.
            *whisper_params:
                WhisperParams in list form (model_size, compute_type, thresholds, etc.).

        Returns:
            Tuple[List[Segment], float]:
                - List of Segment(text, start, end) from the model.
                - Elapsed time in seconds.
        """
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        # Cancel pending offload timer on new request
        with InsanelyFastWhisperInference._model_lock:
            if InsanelyFastWhisperInference._offload_timer and InsanelyFastWhisperInference._offload_timer.is_alive():
                InsanelyFastWhisperInference._offload_timer.cancel()
                logger.debug("Cancelled offload timer due to new transcription request.")

        # Check and update shared model if necessary
        if (
            params.model_size != InsanelyFastWhisperInference._current_model_size
            or InsanelyFastWhisperInference._model is None
            or params.compute_type != InsanelyFastWhisperInference._current_compute_type
        ):
            self.update_model(params.model_size, params.compute_type, progress)

        if progress is not None and callable(progress):
            progress(0, desc="Transcribing...")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
            disable=True,
        ) as rich_progress_display:
            kwargs = {
                "no_speech_threshold": params.no_speech_threshold,
                "temperature": params.temperature,
                "compression_ratio_threshold": params.compression_ratio_threshold,
                "logprob_threshold": params.log_prob_threshold,
            }
            kwargs["language"] = params.lang
            if params.is_translate:
                kwargs["task"] = "translate"

            with InsanelyFastWhisperInference._model_lock:
                if InsanelyFastWhisperInference._model is None:
                    logger.error("Model not loaded. Aborting transcription.")
                    raise RuntimeError("Model is not available for transcription.")
                model_output = InsanelyFastWhisperInference._model(
                    inputs=audio,
                    return_timestamps=True,
                    chunk_length_s=params.chunk_length,
                    batch_size=params.batch_size,
                    generate_kwargs=kwargs,
                )

        segments_result: List[Segment] = []
        if model_output and "chunks" in model_output:
            for item in model_output["chunks"]:
                start, end = item["timestamp"][0], item["timestamp"][1] or item["timestamp"][0]
                segments_result.append(Segment(text=item["text"], start=start, end=end))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: Optional[gr.Progress] = None,
    ):
        """
        Load or switch the underlying Whisper model.

        Args:
            model_size (str): Name of the model to load (e.g. "tiny", "base", "distil-...").
            compute_type (str): Torch dtype (e.g. "float16", "float32") for weights.
            progress (gr.Progress, optional): Progress callback while downloading/loading.

        Raises:
            RuntimeError: If the pipeline download or load fails.
        """
        with InsanelyFastWhisperInference._model_lock:
            if (
                InsanelyFastWhisperInference._model is not None
                and InsanelyFastWhisperInference._current_model_size == model_size
                and InsanelyFastWhisperInference._current_compute_type == compute_type
            ):
                logger.info(f"Model {model_size} with {compute_type} already loaded.")
                if (
                    InsanelyFastWhisperInference._offload_timer
                    and InsanelyFastWhisperInference._offload_timer.is_alive()
                ):
                    InsanelyFastWhisperInference._offload_timer.cancel()
                    logger.debug("Cancelled offload timer as model remains in use.")
                return
            # Offload previous model
            self._perform_actual_offload()
            logger.info("Offloaded previous model during update_model.")

            if progress is not None and callable(progress):
                progress(0, desc="Initializing model...")

            model_path = os.path.join(self.model_dir, model_size)
            if not os.path.isdir(model_path) or not os.listdir(model_path):
                InsanelyFastWhisperInference.download_model(model_size, model_path, progress)

            pipeline_output = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                torch_dtype=compute_type,
                device=self.device,
                model_kwargs=(
                    {"attn_implementation": "flash_attention_2"}
                    if is_flash_attn_2_available()
                    else {"attn_implementation": "sdpa"}
                ),
            )

            if pipeline_output is None:
                logger.error(f"Failed to load model {model_size}.")
                InsanelyFastWhisperInference._model = None
                InsanelyFastWhisperInference._current_model_size = None
                InsanelyFastWhisperInference._current_compute_type = None
                raise RuntimeError(f"Failed to load model '{model_size}': pipeline returned None.")

            InsanelyFastWhisperInference._model = pipeline_output
            InsanelyFastWhisperInference._current_model_size = model_size
            InsanelyFastWhisperInference._current_compute_type = compute_type
            logger.info(f"Loaded model {model_size} with {compute_type} on {self.device}.")

            # Patch generate for float token sanitization
            if hasattr(pipeline_output, "model") and hasattr(pipeline_output.model, "generate"):
                original_generate = pipeline_output.model.generate
                if not hasattr(original_generate, "_is_patched"):

                    def patched_generate(*args, **kwargs):
                        raw = original_generate(*args, **kwargs)

                        def sanitize(t):
                            if isinstance(t, torch.Tensor) and t.is_floating_point():
                                return t.long()
                            return t

                        if isinstance(raw, torch.Tensor):
                            return sanitize(raw)
                        if hasattr(raw, "sequences"):
                            raw.sequences = sanitize(raw.sequences)
                            return raw
                        if isinstance(raw, (list, tuple)):
                            return type(raw)(sanitize(x) if isinstance(x, torch.Tensor) else x for x in raw)
                        return raw

                    patched_generate._is_patched = True
                    pipeline_output.model.generate = patched_generate
                    logger.info("Patched generate for float token sanitization.")

    def _perform_actual_offload(self):
        """
        Immediately delete the in-memory model, clear GPU/XPU caches, and run garbage collection.
        """
        with InsanelyFastWhisperInference._model_lock:
            if InsanelyFastWhisperInference._model is not None:
                del InsanelyFastWhisperInference._model
                InsanelyFastWhisperInference._model = None
                InsanelyFastWhisperInference._current_model_size = None
                InsanelyFastWhisperInference._current_compute_type = None
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "xpu":
                    torch.xpu.empty_cache()
                gc.collect()
                logger.info("Performed actual offload.")
            else:
                logger.info("Model already offloaded.")

    def offload(self, idle_timeout: int = 60):
        """
        Schedule model offload after a period of inactivity.

        Args:
            idle_timeout (int): Seconds to wait before auto-offloading the model.
        """
        with InsanelyFastWhisperInference._model_lock:
            if InsanelyFastWhisperInference._offload_timer and InsanelyFastWhisperInference._offload_timer.is_alive():
                InsanelyFastWhisperInference._offload_timer.cancel()
                logger.debug("Cancelled previous offload timer.")
            InsanelyFastWhisperInference._offload_timer = threading.Timer(idle_timeout, self._perform_actual_offload)
            InsanelyFastWhisperInference._offload_timer.daemon = True
            InsanelyFastWhisperInference._offload_timer.start()
            logger.info(f"Scheduled offload in {idle_timeout} seconds.")

    def get_model_paths(self):
        """
        Enumerate available Whisper model names.

        Returns:
            List[str]: Sorted list of default and locally-downloaded model sizes.
        """
        openai_models = whisper.available_models()
        distil_models = ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"]
        default = openai_models + distil_models
        existing = os.listdir(self.model_dir)
        wrong = [".locks", "insanely_fast_whisper_models_will_be_saved_here"]
        result = [m for m in default + existing if m not in wrong]
        return sorted(set(result), key=(default + existing).index)

    @staticmethod
    def download_model(model_size: str, download_root: str, progress: Optional[gr.Progress] = None):
        """
        Download all required files for a Whisper model from HuggingFace Hub.

        Args:
            model_size (str): Model identifier (e.g. "tiny", "distil-medium.en").
            download_root (str): Local directory to store the model files.
            progress (gr.Progress, optional): Progress callback during download.
        """
        if progress and callable(progress):
            progress(0, desc="Downloading model...")
        os.makedirs(download_root, exist_ok=True)
        files = [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "vocab.json",
        ]
        repo = f"distil-whisper/{model_size}" if model_size.startswith("distil") else f"openai/whisper-{model_size}"
        for f in files:
            hf_hub_download(repo_id=repo, filename=f, local_dir=download_root)
