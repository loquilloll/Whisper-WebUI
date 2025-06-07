import threading
import os
import time
import numpy as np
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import gradio as gr
from huggingface_hub import hf_hub_download
import whisper
import gc
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from argparse import Namespace

from modules.utils.paths import (INSANELY_FAST_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class InsanelyFastWhisperInference(BaseTranscriptionPipeline):
    _model: Optional[pipeline] = None
    _model_lock = threading.Lock()
    _current_model_size: Optional[str] = None
    _current_compute_type: Optional[str] = None

    def __init__(self,
                 model_dir: str = INSANELY_FAST_WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        super().__init__(
            model_dir=model_dir,
            output_dir=output_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.available_models = self.get_model_paths()

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   progress: Optional[gr.Progress] = None,
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: Optional[gr.Progress]
            Indicator to show progress directly in gradio.
        progress_callback: Optional[Callable]
            callback function to show progress in the backend.
        *whisper_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        # Check and update shared model if necessary
        if (params.model_size != InsanelyFastWhisperInference._current_model_size or
                InsanelyFastWhisperInference._model is None or
                params.compute_type != InsanelyFastWhisperInference._current_compute_type):
            self.update_model(params.model_size, params.compute_type, progress)

        # Update Gradio progress if available and callable
        if progress and hasattr(progress, "__call__"):
            progress(0, desc="Transcribing... (Rich console progress is disabled)")

        # Always disable rich.progress.Progress
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(style="yellow1", pulse_style="white"),
                TimeElapsedColumn(),
                disable=True
        ) as rich_progress_display:
            kwargs = {
                "no_speech_threshold": params.no_speech_threshold,
                "temperature": params.temperature,
                "compression_ratio_threshold": params.compression_ratio_threshold,
                "logprob_threshold": params.log_prob_threshold,
            }

            # Use class-level current_model_size for language-specific logic
            if InsanelyFastWhisperInference._current_model_size and InsanelyFastWhisperInference._current_model_size.endswith(".en"):
                pass
            else:
                kwargs["language"] = params.lang
                kwargs["task"] = "translate" if params.is_translate else "transcribe"

            model_output = None
            # Serialize access to the shared model for inference
            with InsanelyFastWhisperInference._model_lock:
                if InsanelyFastWhisperInference._model is None:
                    logger.error("Model not loaded despite update_model call. Aborting transcription.")
                    raise RuntimeError("Model is not available for transcription.")

                model_output = InsanelyFastWhisperInference._model(
                    inputs=audio,
                    return_timestamps=True,
                    chunk_length_s=params.chunk_length,
                    batch_size=params.batch_size,
                    generate_kwargs=kwargs
                )

        segments_result = []
        if model_output:
            for item in model_output["chunks"]:
                start, end = item["timestamp"][0], item["timestamp"][1]
                if end is None:
                    end = start
                segments_result.append(Segment(
                    text=item["text"],
                    start=start,
                    end=end
                ))

        elapsed_time = time.time() - start_time

        # Explicitly delete model_output to help GC
        if 'model_output' in locals():
            del model_output

        # Enhanced cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "xpu":
            torch.xpu.empty_cache()

        gc.collect()

        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: Optional[gr.Progress] = None,
                     ):
        """
        Update shared model settings

        Parameters
        ----------
        model_size: str
            Size of whisper model
        compute_type: str
            Compute type for transcription.
        progress: Optional[gr.Progress]
            Indicator to show progress directly in gradio.
        """
        with InsanelyFastWhisperInference._model_lock:
            # If the requested model is already loaded with correct config, skip
            if (InsanelyFastWhisperInference._model is not None and
                    InsanelyFastWhisperInference._current_model_size == model_size and
                    InsanelyFastWhisperInference._current_compute_type == compute_type):
                logger.info(f"Model {model_size} with {compute_type} is already loaded.")
                return

            # Offload previous shared model if exists
            if InsanelyFastWhisperInference._model is not None:
                del InsanelyFastWhisperInference._model
                InsanelyFastWhisperInference._model = None
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "xpu":
                    torch.xpu.empty_cache()
                gc.collect()
                logger.info("Previous shared model offloaded during update_model.")

            if progress:
                progress(0, desc="Initializing Model..")

            model_path = os.path.join(self.model_dir, model_size)
            if not os.path.isdir(model_path) or not os.listdir(model_path):
                InsanelyFastWhisperInference.download_model(
                    model_size=model_size,
                    download_root=model_path,
                    progress=progress
                )

            InsanelyFastWhisperInference._current_compute_type = compute_type
            InsanelyFastWhisperInference._current_model_size = model_size

            # Load the new shared model
            InsanelyFastWhisperInference._model = pipeline(
                "automatic-speech-recognition",
                model=os.path.join(self.model_dir, model_size),
                torch_dtype=InsanelyFastWhisperInference._current_compute_type,
                device=self.device,
                model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
            )
            logger.info(f"Shared model {model_size} with {compute_type} loaded on {self.device}.")

            # Apply monkey-patch to sanitize float token IDs
            if (InsanelyFastWhisperInference._model is not None and 
                    hasattr(InsanelyFastWhisperInference._model, 'model') and
                    hasattr(InsanelyFastWhisperInference._model.model, 'generate') and 
                    callable(InsanelyFastWhisperInference._model.model.generate)):
                original_generate = InsanelyFastWhisperInference._model.model.generate

                if not hasattr(original_generate, '_is_patched_for_float_tokens'):
                    def patched_generate(*args, **kwargs):
                        raw_output = original_generate(*args, **kwargs)

                        def sanitize_tensor(tensor_item):
                            if isinstance(tensor_item, torch.Tensor) and tensor_item.is_floating_point():
                                logger.debug(f"Sanitizing float tensor to long tensor. Original dtype: {tensor_item.dtype}, Shape: {tensor_item.shape}")
                                return tensor_item.long()
                            return tensor_item

                        if isinstance(raw_output, torch.Tensor):
                            return sanitize_tensor(raw_output)
                        elif isinstance(raw_output, (list, tuple)):
                            sanitized_list = []
                            for item in raw_output:
                                if isinstance(item, torch.Tensor):
                                    sanitized_list.append(sanitize_tensor(item))
                                else:
                                    sanitized_list.append(item)
                            return type(raw_output)(sanitized_list)
                        elif hasattr(raw_output, "sequences") and isinstance(getattr(raw_output, "sequences"), torch.Tensor):
                            logger.debug("Sanitizing 'sequences' attribute of ModelOutput.")
                            raw_output.sequences = sanitize_tensor(raw_output.sequences)
                            return raw_output
                        else:
                            logger.warning(
                                f"Unexpected output type from model.generate: {type(raw_output)}. "
                                f"Output structure (first 200 chars): {str(raw_output)[:200]}. Skipping sanitization."
                            )
                            return raw_output

                    patched_generate._is_patched_for_float_tokens = True
                    InsanelyFastWhisperInference._model.model.generate = patched_generate
                    logger.info("Patched shared self.model.model.generate() in InsanelyFastWhisperInference.")
                else:
                    logger.debug("Shared self.model.model.generate() already patched. Skipping.")
            else:
                logger.warning(
                    "Could not patch shared self.model.model.generate() in InsanelyFastWhisperInference: "
                    "model.generate not found, not callable, or model is None."
                )

    def offload(self):
        """
        Offload the shared model and free up GPU/XPU memory.
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
                logger.info("InsanelyFastWhisperInference shared model offloaded.")
            else:
                logger.info("InsanelyFastWhisperInference shared model already offloaded or not loaded.")

    def get_model_paths(self):
        """
        Get available models from models path including fine-tuned model.

        Returns
        ----------
        Name set of models
        """
        openai_models = whisper.available_models()
        distil_models = ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"]
        default_models = openai_models + distil_models

        existing_models = os.listdir(self.model_dir)
        wrong_dirs = [".locks", "insanely_fast_whisper_models_will_be_saved_here"]

        available_models = default_models + existing_models
        available_models = [model for model in available_models if model not in wrong_dirs]
        available_models = sorted(set(available_models), key=available_models.index)

        return available_models

    @staticmethod
    def download_model(
        model_size: str,
        download_root: str,
        progress: Optional[gr.Progress]
    ):
        if progress:
            progress(0, 'Initializing model..')
        logger.info(f'Downloading {model_size} to "{download_root}"....')

        os.makedirs(download_root, exist_ok=True)
        download_list = [
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

        if model_size.startswith("distil"):
            repo_id = f"distil-whisper/{model_size}"
        else:
            repo_id = f"openai/whisper-{model_size}"
        for item in download_list:
            hf_hub_download(repo_id=repo_id, filename=item, local_dir=download_root)
