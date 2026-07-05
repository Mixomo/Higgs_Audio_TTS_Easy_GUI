from __future__ import annotations

import importlib.util
import gc
import json
import os
import subprocess
import sys
import types
from functools import wraps
from pathlib import Path

import numpy as np
import soundfile as sf

from .model_registry import ASR_MODELS, HIGGS_ASR_MODELS, PROCESSOR_MODELS, WHISPER_MODELS, ensure_asr_model, ensure_model


WHISPER_LANGS = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
}


class ASRManager:
    def __init__(self):
        self.whisper_model = None
        self.whisper_label = None
        self.higgs_model = None
        self.higgs_tokenizer = None
        self.higgs_label = None
        self.higgs_transcribe_batch = None

    def _cleanup_cuda_state(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def unload(self):
        self.whisper_model = None
        self.whisper_label = None
        self.higgs_model = None
        self.higgs_tokenizer = None
        self.higgs_label = None
        self.higgs_transcribe_batch = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _read_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return np.mean(audio, axis=1).astype(np.float32), int(sr)

    def _load_whisper(self, label: str):
        if self.whisper_model is not None and self.whisper_label == label:
            return self.whisper_model
        self.higgs_model = None
        self.higgs_tokenizer = None
        self.higgs_transcribe_batch = None
        from faster_whisper import WhisperModel
        import torch

        model_dir = ensure_asr_model(label)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.whisper_model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
        self.whisper_label = label
        return self.whisper_model

    def _load_higgs(self, label: str):
        if self.higgs_model is not None and self.higgs_label == label:
            return self.higgs_model, self.higgs_tokenizer, self.higgs_transcribe_batch
        self.whisper_model = None
        self.whisper_label = None

        import runpy
        import torch
        from transformers.generation import GenerationConfig
        from transformers import AutoModel, AutoTokenizer

        model_dir = ensure_asr_model(label)
        if not hasattr(GenerationConfig, "generation_kwargs"):
            GenerationConfig.generation_kwargs = {}
        sys.path.insert(0, str(model_dir))
        model = AutoModel.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        if hasattr(model, "generation_config") and not hasattr(model.generation_config, "generation_kwargs"):
            model.generation_config.generation_kwargs = {}
        model_cls = type(model)
        if hasattr(model_cls, "_sample") and not getattr(model_cls._sample, "_higgs_streamer_compat", False):
            original_sample = model_cls._sample

            def _sample_compat(self, *args, **kwargs):
                try:
                    return original_sample(self, *args, **kwargs)
                except TypeError as exc:
                    msg = str(exc)
                    if "streamer" not in msg and "past_key_values_buckets" not in msg:
                        raise
                    kwargs.setdefault("streamer", None)
                    kwargs.setdefault("past_key_values_buckets", None)
                    return original_sample(self, *args, **kwargs)

            _sample_compat._higgs_streamer_compat = True
            model_cls._sample = _sample_compat
        if hasattr(model_cls, "forward") and not getattr(model_cls.forward, "_higgs_generate_kwarg_compat", False):
            original_forward = model_cls.forward

            @wraps(original_forward)
            def _forward_compat(self, *args, **kwargs):
                kwargs.pop("tokenizer", None)
                return original_forward(self, *args, **kwargs)

            _forward_compat._higgs_generate_kwarg_compat = True
            model_cls.forward = _forward_compat
        if hasattr(model, "audio_tower") and hasattr(model.audio_tower, "layers"):
            for layer in model.audio_tower.layers:
                if getattr(layer.forward, "_higgs_whisper_layer_compat", False):
                    continue
                original_layer_forward = layer.forward

                def _make_layer_forward_compat(forward_fn):
                    @wraps(forward_fn)
                    def _layer_forward_compat(
                        self,
                        hidden_states,
                        attention_mask=None,
                        layer_head_mask=None,
                        output_attentions=False,
                        **kwargs,
                    ):
                        output = forward_fn(hidden_states, attention_mask=None, **kwargs)
                        if isinstance(output, tuple):
                            return output
                        return (output, None)

                    _layer_forward_compat._higgs_whisper_layer_compat = True
                    return _layer_forward_compat

                layer.forward = types.MethodType(_make_layer_forward_compat(original_layer_forward), layer)
        if hasattr(model, "layers"):
            for layer in model.layers:
                if getattr(layer.forward, "_higgs_qwen_layer_compat", False):
                    continue
                original_layer_forward = layer.forward

                def _make_qwen_layer_forward_compat(forward_fn):
                    @wraps(forward_fn)
                    def _qwen_layer_forward_compat(self, *args, **kwargs):
                        if "past_key_value" in kwargs and "past_key_values" not in kwargs:
                            kwargs["past_key_values"] = kwargs.pop("past_key_value")
                        output = forward_fn(*args, **kwargs)
                        if isinstance(output, tuple):
                            return output
                        return (output, None)

                    _qwen_layer_forward_compat._higgs_qwen_layer_compat = True
                    return _qwen_layer_forward_compat

                layer.forward = types.MethodType(_make_qwen_layer_forward_compat(original_layer_forward), layer)
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model.audio_out_bos_token_id = tok.convert_tokens_to_ids("<|audio_out_bos|>")
        model.audio_eos_token_id = tok.convert_tokens_to_ids("<|audio_eos|>")

        transcribe_path = model_dir / "transcribe.py"
        if not transcribe_path.exists():
            raise FileNotFoundError(f"Higgs ASR helper not found: {transcribe_path}")
        transcribe_batch = runpy.run_path(str(transcribe_path))["transcribe_batch"]
        processor_dir = ensure_model(PROCESSOR_MODELS["Whisper large-v3 processor"])

        class LocalWhisperProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                from transformers import WhisperProcessor

                return WhisperProcessor.from_pretrained(str(processor_dir), **kwargs)

        transcribe_batch.__globals__["WhisperProcessor"] = LocalWhisperProcessor
        self.higgs_model = model
        self.higgs_tokenizer = tok
        self.higgs_transcribe_batch = transcribe_batch
        self.higgs_label = label
        return model, tok, transcribe_batch

    def _transcribe_subprocess(self, audio_path: str, label: str, language: str) -> tuple[str, str]:
        worker = Path(__file__).with_name("asr_worker.py")
        env = dict(os.environ)
        env["HIGGS_ASR_WORKER"] = "1"
        env["HIGGS_NO_COMPILE"] = "1"
        env.pop("TORCH_LOGS", None)
        cmd = [sys.executable, str(worker), "--audio", audio_path, "--model", label, "--language", language]
        print(f"[asr] Starting {label}: {audio_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            env=env,
        )
        stdout, _ = proc.communicate()
        print(f"[asr] Finished {label} with code {proc.returncode}", flush=True)
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        payload = None
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                payload = json.loads(line)
                break
        if proc.returncode != 0 or not payload or not payload.get("ok"):
            detail = (payload or {}).get("error") or stdout.strip()
            trace = (payload or {}).get("traceback")
            if trace:
                detail = f"{detail}\n{trace}"
            raise RuntimeError(f"Higgs ASR worker failed: {detail}")
        return payload.get("text", ""), payload.get("status", f"{label} transcription complete.")

    def transcribe(self, audio_path: str | None, label: str, language: str = "Auto-detect") -> tuple[str, str]:
        if not audio_path:
            return "", "No audio selected."
        if label not in ASR_MODELS:
            raise ValueError(f"Unknown ASR model: {label}")
        if os.environ.get("HIGGS_ASR_WORKER") != "1":
            return self._transcribe_subprocess(audio_path, label, language)

        if label in WHISPER_MODELS:
            model = self._load_whisper(label)
            lang_code = WHISPER_LANGS.get(language)
            segments, info = model.transcribe(audio_path, beam_size=5, language=lang_code)
            text = "".join(seg.text for seg in segments).strip()
            detected = getattr(info, "language", None) or "unknown"
            return text, f"{label} transcription complete. Detected language: {detected}"

        if label in HIGGS_ASR_MODELS:
            self._cleanup_cuda_state()
            audio, sr = self._read_audio(audio_path)
            model, tok, transcribe_batch = self._load_higgs(label)
            texts = transcribe_batch(model, tok, [audio], sample_rates=sr, enable_thinking=False, max_new_tokens=256)
            return (texts[0] if texts else "").strip(), f"{label} transcription complete."

        raise ValueError(f"Unsupported ASR model: {label}")

    def transcribe_whisper_direct(
        self,
        audio_path: str,
        label: str,
        language: str = "Auto-detect",
        batch_size: int = 8,
    ) -> tuple[str, str]:
        if label not in WHISPER_MODELS:
            return self.transcribe(audio_path, label, language)
        model = self._load_whisper(label)
        lang_code = WHISPER_LANGS.get(language)
        batch_size = max(int(batch_size or 1), 1)
        if batch_size > 1:
            from faster_whisper import BatchedInferencePipeline

            pipeline = BatchedInferencePipeline(model=model)
            segments, info = pipeline.transcribe(audio_path, beam_size=5, language=lang_code, batch_size=batch_size)
        else:
            segments, info = model.transcribe(audio_path, beam_size=5, language=lang_code)
        text = "".join(seg.text for seg in segments).strip()
        detected = getattr(info, "language", None) or "unknown"
        return text, f"{label} transcription complete. Language: {language if lang_code else 'auto'} / detected: {detected}"

    def transcribe_many_whisper(
        self,
        audio_paths: list[str],
        label: str,
        language: str = "Auto-detect",
        batch_size: int = 8,
        progress_cb=None,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        total = len(audio_paths)
        for idx, audio_path in enumerate(audio_paths, 1):
            if progress_cb:
                progress_cb(idx - 1, total, f"Transcribing {Path(audio_path).name}")
            text, status = self.transcribe_whisper_direct(audio_path, label, language, batch_size=batch_size)
            print(f"[dataset-asr] {Path(audio_path).name}: {status}", flush=True)
            results[audio_path] = text
            if progress_cb:
                progress_cb(idx, total, f"Transcribed {Path(audio_path).name}")
        return results
