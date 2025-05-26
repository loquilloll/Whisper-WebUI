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

from modules.utils.paths import (INSANELY_FAST_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import Segment, Word, WhisperParams
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class InsanelyFastWhisperInference(BaseTranscriptionPipeline):
    def __init__(
        self,
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

    def transcribe(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.
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

        # Ensure the correct model is loaded
        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        # Signal start of transcription
        if progress_callback:
            progress_callback(0.0)
        else:
            progress(0.0, desc="Transcribing... (InsanelyFastWhisper)")

        # Prepare kwargs for pipeline
        generate_kwargs = {
            "no_speech_threshold": params.no_speech_threshold,
            "temperature": params.temperature,
            "compression_ratio_threshold": params.compression_ratio_threshold,
            "logprob_threshold": params.log_prob_threshold,
        }

        if self.current_model_size.endswith(".en"):
            # English-only models may not require language/task parameters
            pass
        else:
            generate_kwargs["language"] = params.lang
            generate_kwargs["task"] = "translate" if params.is_translate else "transcribe"

        # Determine the correct return_timestamps argument for the pipeline
        pipeline_return_timestamps_arg = "word" if params.word_timestamps else True

        # Run the Hugging Face pipeline
        hf_output = self.model(
            inputs=audio,
            return_timestamps=pipeline_return_timestamps_arg,
            chunk_length_s=params.chunk_length,
            batch_size=params.batch_size,
            generate_kwargs=generate_kwargs
        )

        # Process output into Segment objects
        segments_result: List[Segment] = []
        for item in hf_output.get("chunks", []):
            start, end = item["timestamp"][0], item["timestamp"][1]
            if end is None:
                end = start

            word_list: List[Word] = []
            if params.word_timestamps and item.get("words"):
                for w in item["words"]:
                    word_list.append(Word(
                        word=w.get("text", "").strip(),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        score=w.get("probability", w.get("score", 0.0))
                    ))

            segments_result.append(Segment(
                text=item.get("text", ""),
                start=start,
                end=end,
                words=word_list if word_list else None
            ))

        elapsed_time = time.time() - start_time

        # Signal completion
        if progress_callback:
            progress_callback(1.0)
        else:
            progress(1.0, desc="Completed")

        return segments_result, elapsed_time

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
    ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model
        compute_type: str
            Compute type for transcription.
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Model..")
        model_path = os.path.join(self.model_dir, model_size)
        if not os.path.isdir(model_path) or not os.listdir(model_path):
            self.download_model(
                model_size=model_size,
                download_root=model_path,
                progress=progress
            )

        self.current_compute_type = compute_type
        self.current_model_size = model_size
        self.model = pipeline(
            "automatic-speech-recognition",
            model=os.path.join(self.model_dir, model_size),
            torch_dtype=self.current_compute_type,
            device=self.device,
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )

    def get_model_paths(self) -> List[str]:
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
        progress: gr.Progress
    ):
        progress(0, desc='Initializing model..')
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
