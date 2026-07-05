from __future__ import annotations

import gc
import io
import importlib.util
import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_utils import SAMPLE_RATE, concatenate, split_long_text
from .adapters import NONE_ADAPTER, resolve_v2_lora_adapter, resolve_v3_lora_adapter
from .model_registry import TOKENIZER_MODELS, ensure_model, ensure_tts_model
from .paths import ROOT, V2_DEFAULT_MODEL, V2_DEFAULT_TOKENIZER, V3_MODEL_PATH


def _ensure_local_hf_modules_cache() -> None:
    local_cache = ROOT / "models" / ".cache" / "hf_modules"
    current = os.environ.get("HF_MODULES_CACHE")
    if current:
        try:
            current_path = Path(current)
            current_path.mkdir(parents=True, exist_ok=True)
            probe = current_path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return
        except Exception:
            pass
    local_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(local_cache)


_HIGGS_V3_MISSING_AUDIO_HEAD_RE = re.compile(
    r"\[transformers\][\s\S]*?LOAD REPORT[\s\S]*?"
    r"audio_head\.weight[\s\S]*?"
    r"Notes:\s*\n"
    r"(?:- MISSING:[^\n]*(?:\n|$))+",
    re.MULTILINE,
)


def _apply_torch_threads(torch) -> None:
    try:
        threads = max(1, int(os.environ.get("HIGGS_CPU_THREADS", "8")))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(4, threads // 2)))
    except Exception:
        pass
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _filtered_higgs_v3_load_output(text: str) -> str:
    if "audio_head.weight" not in text or "MISSING" not in text:
        return text
    filtered = _HIGGS_V3_MISSING_AUDIO_HEAD_RE.sub("", text)
    lines = []
    skip_next_rule = False
    for line in filtered.splitlines(keepends=True):
        plain = _ANSI_RE.sub("", line)
        if "HiggsMultimodalQwen3ForConditionalGeneration LOAD REPORT" in plain:
            continue
        if "audio_head.weight" in plain or "newly initialized because missing from the checkpoint" in plain:
            continue
        if "Key" in plain and "Status" in plain:
            continue
        if "------------------" in plain and "---------" in plain:
            continue
        if plain.strip() == "Notes:":
            skip_next_rule = True
            continue
        if skip_next_rule and plain.lstrip().startswith("- MISSING:"):
            skip_next_rule = False
            continue
        skip_next_rule = False
        lines.append(line)
    return "".join(lines)


def _run_with_higgs_v3_load_report_filter(fn):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with _ignore_higgs_v3_missing_audio_head(), redirect_stdout(stdout), redirect_stderr(stderr):
        result = fn()
    for captured, stream in ((stdout.getvalue(), sys.stdout), (stderr.getvalue(), sys.stderr)):
        filtered = _filtered_higgs_v3_load_output(captured)
        if filtered:
            stream.write(filtered)
            stream.flush()
    return result


@contextmanager
def _ignore_higgs_v3_missing_audio_head():
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        yield
        return

    previous = getattr(PreTrainedModel, "_keys_to_ignore_on_load_missing", None)
    patterns = set(previous or [])
    patterns.add(r"^audio_head\.weight$")
    PreTrainedModel._keys_to_ignore_on_load_missing = patterns
    try:
        yield
    finally:
        PreTrainedModel._keys_to_ignore_on_load_missing = previous


class HiggsV3Engine:
    REF_MAX_SEC = 30

    def __init__(self, model_path: str | Path = V3_MODEL_PATH):
        self.model_path = str(model_path)
        self.model = None
        self.tokenizer = None
        self.cancel_requested = False
        self.ref_cache: dict[str, tuple[float, object, bool]] = {}
        self.forced_precision = "auto"
        self.attention_backend = "sdpa"
        self.frames_per_sec: float | None = None
        self.adapter_choice: str | None = None
        self.loaded_adapter_path: str | None = None
        self.compiled = False
        self.compile_enabled = False
        self.last_generation_frames = 0
        self.last_generation_audio_sec = 0.0

    def set_runtime(
        self,
        precision: str = "auto",
        attention_backend: str | None = None,
        compile_enabled: bool | None = None,
    ):
        precision = precision if precision in {"auto", "bf16", "8bit", "4bit"} else "auto"
        attention_backend = attention_backend if attention_backend in {"sdpa", "eager"} else self.attention_backend
        compile_value = self.compile_enabled if compile_enabled is None else bool(compile_enabled)
        if precision != self.forced_precision or attention_backend != self.attention_backend or compile_value != self.compile_enabled:
            self.forced_precision = precision
            self.attention_backend = attention_backend
            self.compile_enabled = compile_value
            self.unload()

    def set_adapter(self, adapter_choice: str | None):
        normalized = adapter_choice or "None"
        if normalized != (self.adapter_choice or "None"):
            self.adapter_choice = normalized
            self.unload()

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        _ensure_local_hf_modules_cache()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        _apply_torch_threads(torch)
        t0 = time.perf_counter()
        model_dir = ensure_tts_model("Higgs V3 TTS")
        device = self._detect_device(torch)
        precision = self._precision(device)
        quant = None
        skip = ["audio_head", "audio_embedding"]
        if device == "cuda" and precision == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip,
            )
        elif device == "cuda" and precision == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "mps":
            dtype = torch.float16
        kwargs = {"trust_remote_code": True, "dtype": dtype}
        if quant is not None:
            kwargs["quantization_config"] = quant
            kwargs["device_map"] = "auto"
        try:
            self.model = _run_with_higgs_v3_load_report_filter(
                lambda: AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    attn_implementation=self.attention_backend,
                    **kwargs,
                )
            )
        except Exception:
            try:
                self.model = _run_with_higgs_v3_load_report_filter(
                    lambda: AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
                )
            except Exception:
                raise
        tokenizer_dir = ensure_model(TOKENIZER_MODELS["Higgs V2 tokenizer"])
        if hasattr(self.model, "config") and hasattr(self.model.config, "audio_tokenizer_id"):
            self.model.config.audio_tokenizer_id = str(tokenizer_dir)
        if quant is None and device not in {"cpu"}:
            self.model = self.model.to(device)
        self._maybe_load_adapter()
        self.model.eval()
        try:
            self.model.get_audio_codec()
        except Exception:
            pass
        if device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        self._maybe_compile(torch, device, precision)
        print(
            "[v3-load] "
            f"device={device} precision={precision} attention={self.attention_backend} "
            f"adapter={self.adapter_choice or 'None'} "
            f"compile={'on' if self.compiled else 'off'} "
            f"time={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def _maybe_compile(self, torch, device: str, precision: str):
        self.compiled = False
        if device != "cuda" or precision != "bf16":
            return
        if not self.compile_enabled:
            return
        if os.environ.get("HIGGS_NO_COMPILE", "").strip().lower() in {"1", "true", "yes"}:
            return
        try:
            import triton  # noqa: F401
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            self.model.model = torch.compile(self.model.model, dynamic=True)
            self.compiled = True
            print("[v3-load] torch.compile dynamic enabled.", flush=True)
        except ModuleNotFoundError as exc:
            if exc.name == "triton":
                print("[v3-load] torch.compile disabled: Triton is not installed. Run install.bat with a CUDA backend.", flush=True)
                return
            print(f"[v3-load] torch.compile unavailable; running uncompiled with selected attention backend ({exc})", flush=True)
        except Exception as exc:
            print(f"[v3-load] torch.compile unavailable; running uncompiled with selected attention backend ({exc})", flush=True)

    def _maybe_load_adapter(self):
        adapter_path = resolve_v3_lora_adapter(self.adapter_choice)
        self.loaded_adapter_path = None
        if adapter_path is None:
            return
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError(f"PEFT is required to load V3 LoRA adapters: {exc}") from exc
        print(f"[v3-adapter] Loading LoRA adapter: {adapter_path}", flush=True)
        self.model.model = PeftModel.from_pretrained(self.model.model, str(adapter_path))
        self.loaded_adapter_path = str(adapter_path)

    def _precision(self, device: str) -> str:
        if device != "cuda":
            return "cpu"
        pick = self.forced_precision if self.forced_precision != "auto" else os.environ.get("HIGGS_TTS_PRECISION", "").strip().lower()
        if pick in {"bf16", "8bit", "4bit"}:
            return pick
        return "bf16"

    def _detect_device(self, torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        try:
            import torch_directml

            return str(torch_directml.device())
        except Exception:
            return "cpu"

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.loaded_adapter_path = None
        self.compiled = False
        self.ref_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def cancel(self):
        self.cancel_requested = True

    def _load_ref(self, path: str):
        import soundfile as sf
        import torch

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data).mean(dim=1), sr

    def _ref_codes(self, path: str):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        hit = self.ref_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1], hit[2]
        wav, sr = self._load_ref(path)
        cap = int(sr * self.REF_MAX_SEC)
        trimmed = wav.shape[-1] > cap
        if trimmed:
            wav = wav[:cap]
        codes = self.model._encode_reference(wav, sr).cpu()
        self.ref_cache[path] = (mtime, codes, trimmed)
        return codes, trimmed

    def _modeling_module(self):
        return sys.modules[type(self.model).__module__]

    def _generate_stream(
        self,
        text: str,
        reference_codes=None,
        reference_text: str = "",
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 2048,
        progress_cb=None,
    ):
        import torch

        try:
            mod = self._modeling_module()
            apply_delay_pattern = mod.apply_delay_pattern
            reverse_delay_pattern = mod.reverse_delay_pattern
            sampler_state = mod._SamplerState
            sampler_step = mod._sampler_step
            num_codebooks = self.model.num_codebooks
            _ = (
                self.model._build_prompt_ids,
                self.model._prefill_embeds,
                self.model._decode_codes,
                self.model.audio_head,
                self.model.audio_embedding,
                self.model.model,
            )
        except Exception:
            return None

        with torch.no_grad():
            delayed_ref = apply_delay_pattern(reference_codes.to(torch.long)) if reference_codes is not None else None
            prompt_ids = self.model._build_prompt_ids(
                self.tokenizer,
                text,
                num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
                reference_text=reference_text or None,
            )
            inputs_embeds = self.model._prefill_embeds(prompt_ids, delayed_ref)
            out = self.model.model(inputs_embeds=inputs_embeds, use_cache=True)
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position = inputs_embeds.shape[1]
            state = sampler_state(num_codebooks=num_codebooks)
            rows = []
            last_bar_step = 0
            try:
                from tqdm.auto import tqdm

                bar = tqdm(total=int(max_new_tokens), unit="frame", desc="[gen] Higgs V3", dynamic_ncols=True, leave=True)
            except Exception:
                bar = None
            try:
                for step in range(int(max_new_tokens)):
                    if self.cancel_requested:
                        return torch.zeros(0, dtype=torch.float32)
                    logits = self.model.audio_head(hidden_last).to(torch.float32)[0]
                    codes = sampler_step(logits, state, temperature=temperature, top_p=top_p, top_k=top_k)
                    if state.generation_done:
                        break
                    rows.append(codes.cpu())
                    if progress_cb is not None and (step == 0 or step % 8 == 0):
                        try:
                            approx_sec = len(rows) / float(self.frames_per_sec or 25.0)
                            progress_cb(
                                step + 1,
                                int(max_new_tokens),
                                f"[gen] Higgs V3 {step + 1}/{int(max_new_tokens)} frames, ~{approx_sec:.1f}s audio",
                            )
                        except Exception:
                            pass
                    if bar is not None:
                        if step == 0 or (step + 1) % 8 == 0:
                            bar.update(step + 1 - last_bar_step)
                            last_bar_step = step + 1
                        if step % 32 == 0:
                            bar.set_postfix_str(f"~{len(rows) / float(self.frames_per_sec or 25.0):.1f}s audio")
                    elif step and step % 64 == 0:
                        print(f"[gen] Higgs V3 {step}/{max_new_tokens} frames", flush=True)
                    step_embed = self.model.audio_embedding(codes.unsqueeze(0)).unsqueeze(1)
                    cache_pos = torch.tensor([position], device=self.model.device)
                    out = self.model.model(
                        inputs_embeds=step_embed.to(inputs_embeds.dtype),
                        past_key_values=past,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    past = out.past_key_values
                    hidden_last = out.last_hidden_state[:, -1, :]
                    position += 1
            finally:
                if bar is not None:
                    if len(rows) > last_bar_step:
                        bar.update(len(rows) - last_bar_step)
                    bar.close()
                    print("", flush=True)
            if len(rows) < num_codebooks:
                return torch.zeros(0, dtype=torch.float32)
            delayed = torch.stack(rows, dim=0)
            codes = reverse_delay_pattern(delayed)
            audio = self.model._decode_codes(codes)
            sec = float(audio.shape[-1]) / float(SAMPLE_RATE) if getattr(audio, "shape", None) is not None else 0.0
            if sec > 0.05:
                self.frames_per_sec = len(rows) / sec
            self.last_generation_frames = len(rows)
            self.last_generation_audio_sec = sec
            return audio

    def generate(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        seed: int = -1,
        progress_cb=None,
    ) -> tuple[int, np.ndarray, str]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        self.cancel_requested = False
        self.last_generation_frames = 0
        self.last_generation_audio_sec = 0.0
        t0 = time.perf_counter()
        self.load()
        import torch

        _apply_torch_threads(torch)
        if seed is not None and int(seed) >= 0:
            torch.manual_seed(int(seed))
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": max(float(temperature), 0.3),
            "top_p": float(top_p) if float(top_p) < 1.0 else None,
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "progress_cb": progress_cb,
        }
        trimmed = False
        if ref_audio:
            ref_codes, trimmed = self._ref_codes(ref_audio)
            kwargs["reference_codes"] = ref_codes
            if ref_text and not trimmed:
                kwargs["reference_text"] = ref_text.strip()
        deterministic = False
        audio = self._generate_stream(text, **kwargs)
        if audio is None:
            try:
                audio = self.model.generate_speech(text, self.tokenizer, **kwargs)
            except TypeError:
                kwargs.pop("progress_cb", None)
                kwargs.pop("reference_codes", None)
                if ref_audio:
                    ref_wav, ref_sr = self._load_ref(ref_audio)
                    kwargs["reference_audio"] = ref_wav
                    kwargs["reference_sample_rate"] = ref_sr
                    if ref_text:
                        kwargs["reference_text"] = ref_text.strip()
                audio = self.model.generate_speech(text, self.tokenizer, **kwargs)
        elif self.cancel_requested:
            return SAMPLE_RATE, np.zeros(0, dtype=np.float32), "V3 generation cancelled."
        if audio is None:
            kwargs.pop("progress_cb", None)
            kwargs.pop("reference_codes", None)
            audio = self.model.generate_speech(text, self.tokenizer, **kwargs)
        if self.cancel_requested:
            return SAMPLE_RATE, np.zeros(0, dtype=np.float32), "V3 generation cancelled."
        elapsed = time.perf_counter() - t0
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        sec = len(audio_np) / SAMPLE_RATE if len(audio_np) else 0.0
        frames = self.last_generation_frames or int(sec * (self.frames_per_sec or 0.0))
        print(f"[gen] Higgs V3 OK: {frames} frames -> {sec:.1f}s audio", flush=True)
        diagnostics = (
            f"[v3-gen-diagnostics] precision={self.forced_precision} "
            f"text_chars={len(text)} max_new_tokens={max_new_tokens} "
            f"audio_sec={sec:.2f} elapsed={elapsed:.2f}s rtf={elapsed / max(sec, 1e-6):.2f}"
        )
        return SAMPLE_RATE, audio_np, (
            "V3 local Transformers generation complete."
            + f"\n{diagnostics}"
            + (f" Adapter: {self.loaded_adapter_path}." if self.loaded_adapter_path else "")
            + (" Reference was trimmed to 30s." if trimmed else "")
        )

class HiggsV2Engine:
    def __init__(self, model_path: str = V2_DEFAULT_MODEL, tokenizer_path: str = V2_DEFAULT_TOKENIZER):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.client = None
        self.model = None
        self.processor = None
        self.audio_tokenizer = None
        self.generation = None
        self.adapter_choice: str | None = None
        self.loaded_adapter_path: str | None = None

    def set_adapter(self, adapter_choice: str | None):
        normalized = adapter_choice or NONE_ADAPTER
        if normalized != (self.adapter_choice or NONE_ADAPTER):
            self.adapter_choice = normalized
            self.unload()

    @staticmethod
    def _normalize_event_tags(text: str) -> str:
        for tag, replacement in [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE_s>[Humming]</SE_s>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]:
            text = text.replace(tag, replacement)
        return text

    def _load_generation_module(self):
        if self.generation is not None:
            return self.generation
        gen_path = ROOT / "train-higgs-audio" / "examples" / "generation.py"
        if not gen_path.exists():
            raise FileNotFoundError(f"V2 generation script not found: {gen_path}")
        self._patch_transformers_llama_attention()
        train_root = str(ROOT / "train-higgs-audio")
        if train_root not in sys.path:
            sys.path.insert(0, train_root)
        spec = importlib.util.spec_from_file_location("higgs_v2_generation", gen_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        self.generation = module
        return module

    def _patch_transformers_llama_attention(self):
        try:
            import transformers.models.llama.modeling_llama as llama

            if not hasattr(llama, "LLAMA_ATTENTION_CLASSES") and hasattr(llama, "LlamaAttention"):
                llama.LLAMA_ATTENTION_CLASSES = {
                    "eager": llama.LlamaAttention,
                    "sdpa": llama.LlamaAttention,
                }
        except Exception:
            pass

    def _load_audio_tokenizer(self, tokenizer_path: str, device: str):
        from pathlib import Path

        tok_dir = Path(tokenizer_path)
        if (tok_dir / "model.safetensors").exists():
            import librosa
            import torch
            from transformers import AutoModel

            class NativeV2AudioTokenizer:
                def __init__(self, path: str, device: str):
                    self.model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
                    self.device = device
                    self.sampling_rate = int(getattr(self.model.config, "sample_rate", 24000))
                    self.tps = self.sampling_rate // int(getattr(self.model.config, "downsample_factor", 960))
                    acoustic_config = getattr(self.model.config, "acoustic_model_config", {}) or {}
                    if isinstance(acoustic_config, dict):
                        self.num_codebooks = int(acoustic_config.get("n_codebooks", 8))
                    else:
                        self.num_codebooks = int(getattr(acoustic_config, "n_codebooks", 8))
                    self.codebook_size = int(getattr(self.model.config, "codebook_size", 1024))

                def encode(self, audio_path_or_wv, sr=None):
                    if isinstance(audio_path_or_wv, str):
                        wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
                    else:
                        wv = np.asarray(audio_path_or_wv, dtype=np.float32)
                        if sr is None:
                            sr = self.sampling_rate
                    if int(sr) != self.sampling_rate:
                        wv = librosa.resample(wv, orig_sr=int(sr), target_sr=self.sampling_rate)
                    x = torch.from_numpy(np.asarray(wv, dtype=np.float32)).to(self.device).view(1, 1, -1)
                    with torch.inference_mode():
                        out = self.model.encode(x)
                    codes = getattr(out, "audio_codes", out)
                    return codes[0].detach().cpu()

                def decode(self, audio_codes):
                    codes = audio_codes.to(self.device)
                    with torch.inference_mode():
                        out = self.model.decode(codes)
                    audio = getattr(out, "audio_values", out)
                    return audio.detach().cpu().numpy()

            return NativeV2AudioTokenizer(str(tok_dir), device)

        gen = self._load_generation_module()
        return gen.load_higgs_audio_tokenizer(tokenizer_path, device=device)

    def load(self, max_new_tokens: int = 2048):
        if self.model is not None and self.processor is not None:
            return
        if self.client is not None:
            return
        import torch

        _apply_torch_threads(torch)
        device_id = 0 if torch.cuda.is_available() else None
        device = f"cuda:{device_id}" if device_id is not None else "cpu"
        model_path = self.model_path
        tokenizer_path = self.tokenizer_path
        if model_path == V2_DEFAULT_MODEL:
            model_path = str(ensure_tts_model("Higgs V2 TTS"))
        if tokenizer_path == V2_DEFAULT_TOKENIZER:
            tokenizer_path = str(ensure_model(TOKENIZER_MODELS["Higgs V2 tokenizer"]))
        try:
            from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

            processor_kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                processor_kwargs["device_map"] = "auto"
            self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            kwargs = {"dtype": dtype}
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
            self.model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()
            if not torch.cuda.is_available():
                self.model = self.model.to(device)
            self._maybe_load_adapter()
            return
        except Exception as exc:
            if self.adapter_choice and self.adapter_choice != NONE_ADAPTER:
                raise RuntimeError(f"V2 LoRA adapter requires the native Transformers backend; native load failed: {exc}") from exc
            print(f"[v2] Native Transformers load failed, falling back to legacy loader: {exc}", flush=True)
        gen = self._load_generation_module()
        self.audio_tokenizer = self._load_audio_tokenizer(tokenizer_path, device=device)
        self.client = gen.HiggsAudioModelClient(
            model_path=model_path,
            audio_tokenizer=self.audio_tokenizer,
            device_id=device_id,
            max_new_tokens=int(max_new_tokens),
            use_static_kv_cache=bool(torch.cuda.is_available()),
        )

    def unload(self):
        self.client = None
        self.model = None
        self.processor = None
        self.audio_tokenizer = None
        self.loaded_adapter_path = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _maybe_load_adapter(self):
        adapter_path = resolve_v2_lora_adapter(self.adapter_choice)
        self.loaded_adapter_path = None
        if adapter_path is None:
            return
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError(f"PEFT is required to load V2 LoRA adapters: {exc}") from exc
        print(f"[v2-adapter] Loading LoRA adapter: {adapter_path}", flush=True)
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path)).eval()
        self.loaded_adapter_path = str(adapter_path)
        active = getattr(self.model, "active_adapter", None)
        print(f"[v2-adapter] Active adapter: {active or 'default'}", flush=True)

    def _messages_for(self, text: str, ref_audio: Optional[str], ref_text: str):
        gen = self._load_generation_module()
        speaker_tags = sorted(set(__import__("re").findall(r"\[(SPEAKER\d+)\]", text or "")))
        if ref_audio:
            audio_ids = [self.audio_tokenizer.encode(ref_audio)]
            messages = [
                gen.Message(
                    role="system",
                    content=(
                        "Generate audio following instruction.\n\n"
                        "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
                    ),
                ),
                gen.Message(role="user", content=(ref_text or "This is a voice reference.").strip()),
                gen.Message(role="assistant", content=gen.AudioContent(audio_url=ref_audio)),
            ]
            return messages, audio_ids
        return gen.prepare_generation_context(
            scene_prompt="Audio is recorded from a quiet room.",
            ref_audio=None,
            ref_audio_in_system_message=False,
            audio_tokenizer=self.audio_tokenizer,
            speaker_tags=speaker_tags,
        )

    def _native_conversation(self, text: str, ref_audio: Optional[str], ref_text: str):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Generate audio following instruction."}],
            },
            {
                "role": "scene",
                "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}],
            },
        ]
        if ref_audio:
            conversation.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (ref_text or "This is a voice reference.").strip()}],
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "audio", "url": ref_audio}],
                }
            )
        conversation.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return conversation

    def _native_generate(
        self,
        text: str,
        ref_audio: Optional[str],
        ref_text: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        ras_win_len: int,
        ras_win_max_num_repeat: int,
        seed: int,
        progress_cb=None,
    ) -> tuple[int, np.ndarray, str]:
        import tempfile
        import torch
        import soundfile as sf
        from transformers import StoppingCriteria, StoppingCriteriaList

        _apply_torch_threads(torch)
        if seed is not None and int(seed) >= 0:
            torch.manual_seed(int(seed))
        conversation = self._native_conversation(text, ref_audio, ref_text)
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"sampling_rate": 24000},
            return_tensors="pt",
        ).to(self.model.device)
        adapter_voice_mode = bool(self.loaded_adapter_path and not ref_audio)
        del adapter_voice_mode
        temperature = max(float(temperature), 0.3)
        max_new_tokens = int(max_new_tokens)
        prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

        def _bar(step: int, total: int) -> str:
            width = 28
            total = max(int(total), 1)
            step = min(max(int(step), 0), total)
            filled = int(width * step / total)
            return "█" * filled + "░" * (width - filled)

        class _GenerationProgress(StoppingCriteria):
            def __init__(self) -> None:
                self.last_reported = -1

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                step = max(int(input_ids.shape[-1]) - prompt_len, 0)
                if step == 0 or step >= max_new_tokens or step - self.last_reported >= 8:
                    self.last_reported = step
                    pct = 100.0 * step / max(max_new_tokens, 1)
                    desc = f"[gen] Higgs V2 {step}/{max_new_tokens} audio frames/tokens, ~{step / 25.0:.1f}s audio"
                    print(
                        f"[gen] Higgs V2 |{_bar(step, max_new_tokens)}| {pct:5.1f}% "
                        f"{step}/{max_new_tokens} audio frames/tokens, ~{step / 25.0:.1f}s audio",
                        flush=True,
                    )
                    if progress_cb is not None:
                        try:
                            progress_cb(step, max_new_tokens, desc)
                        except Exception:
                            pass
                return False

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_k": int(top_k),
            "top_p": float(top_p),
            "ras_win_len": None if int(ras_win_len) <= 0 else int(ras_win_len),
            "ras_win_max_num_repeat": int(ras_win_max_num_repeat),
            "stopping_criteria": StoppingCriteriaList([_GenerationProgress()]),
        }
        if progress_cb is not None:
            progress_cb(0, max_new_tokens, f"[gen] Higgs V2 0/{max_new_tokens} audio frames/tokens, ~0.0s audio")
        outputs = self.model.generate(**inputs, **generation_kwargs)
        if progress_cb is not None:
            progress_cb(max_new_tokens, max_new_tokens, f"[gen] Higgs V2 decoding generated audio")
        decoded = self.processor.batch_decode(outputs)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp = handle.name
        try:
            self.processor.save_audio(decoded, tmp)
            wav, sr = sf.read(tmp, dtype="float32", always_2d=False)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        duration = float(np.asarray(wav).shape[0]) / float(sr) if sr else 0.0
        print(f"[gen] Higgs V2 OK: {duration:.1f}s audio", flush=True)
        return int(sr), np.asarray(wav, dtype=np.float32), (
            f"V2 native Transformers generation complete. Audio: {duration:.1f}s."
            + (f" Adapter: {self.loaded_adapter_path}." if self.loaded_adapter_path else "")
        )

    def generate(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2,
        seed: int = -1,
        chunk_mode: str = "None",
        progress_cb=None,
    ) -> tuple[int, np.ndarray, str]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        text = self._normalize_event_tags(text)
        self.load(max_new_tokens=max_new_tokens)
        if self.model is not None and self.processor is not None:
            if not any(text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
                text += "."
            return self._native_generate(
                text,
                ref_audio,
                ref_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                ras_win_len,
                ras_win_max_num_repeat,
                seed,
                progress_cb=progress_cb,
            )
        gen = self._load_generation_module()
        text = gen.normalize_chinese_punctuation(text).replace("(", " ").replace(")", " ")
        if not any(text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
            text += "."
        messages, audio_ids = self._messages_for(text, ref_audio, ref_text)
        method = {"None": None, "Paragraphs": "word", "Lines": "word", "Speaker turns": "speaker"}.get(chunk_mode, None)
        chunked = gen.prepare_chunk_text(text, chunk_method=method, chunk_max_word_num=180, chunk_max_num_turns=1)
        wav, sr, generated_text = self.client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked,
            generation_chunk_buffer_size=None,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            ras_win_len=None if int(ras_win_len) <= 0 else int(ras_win_len),
            ras_win_max_num_repeat=int(ras_win_max_num_repeat),
            seed=None if seed is None or int(seed) < 0 else int(seed),
        )
        return int(sr), np.asarray(wav, dtype=np.float32), generated_text or "V2 generation complete."


class ModelManager:
    def __init__(self):
        self.v3 = HiggsV3Engine()
        self.v2 = HiggsV2Engine()
        self.active = None

    def set_v2_paths(self, model_path: str, tokenizer_path: str):
        model_path = (model_path or V2_DEFAULT_MODEL).strip()
        tokenizer_path = (tokenizer_path or V2_DEFAULT_TOKENIZER).strip()
        if model_path != self.v2.model_path or tokenizer_path != self.v2.tokenizer_path:
            adapter_choice = self.v2.adapter_choice
            self.v2.unload()
            self.v2 = HiggsV2Engine(model_path=model_path, tokenizer_path=tokenizer_path)
            self.v2.set_adapter(adapter_choice)

    def set_v3_runtime(
        self,
        precision: str,
        attention_backend: str | None = None,
        compile_enabled: bool | None = None,
    ):
        self.v3.set_runtime(precision, attention_backend, compile_enabled)

    def set_v3_adapter(self, adapter_choice: str | None):
        self.v3.set_adapter(adapter_choice)

    def set_lora_adapter(self, adapter_choice: str | None):
        normalized = adapter_choice or NONE_ADAPTER
        self.v2.set_adapter(normalized)
        self.v3.set_adapter(normalized)

    def unload_other(self, version: str):
        if version.startswith("Higgs V3"):
            self.v2.unload()
            self.active = "v3"
        else:
            self.v3.unload()
            self.active = "v2"

    def generate(self, version: str, **kwargs) -> tuple[int, np.ndarray, str]:
        self.unload_other(version)
        if version.startswith("Higgs V3"):
            return self.v3.generate(
                **{k: v for k, v in kwargs.items() if k not in {"chunk_mode", "ras_win_len", "ras_win_max_num_repeat"}}
            )
        return self.v2.generate(**kwargs)

    def generate_many(
        self,
        version: str,
        chunks: list[dict],
        gap_seconds: float,
        chunk_progress=None,
        **kwargs,
    ) -> tuple[int, np.ndarray, str]:
        audios = []
        logs = []
        sr = SAMPLE_RATE
        for idx, chunk in enumerate(chunks, 1):
            if chunk_progress:
                chunk_progress(idx, len(chunks), chunk.get("text", ""), chunk)
            local = dict(kwargs)
            local.update(chunk)
            local = {k: v for k, v in local.items() if not str(k).startswith("_")}
            sr, wav, log = self.generate(version, **local)
            audios.append(wav)
            logs.append(f"{idx}. {log}")
        return sr, concatenate(audios, sr=sr, gap_seconds=gap_seconds), "\n".join(logs)

    def unload_all(self):
        self.v2.unload()
        self.v3.unload()
        self.active = None


def env_summary() -> str:
    lines = [f"Root: {ROOT}", f"V3 model: {V3_MODEL_PATH}"]
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            lines.append(f"CUDA: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            lines.append("CUDA: not available")
    except Exception as exc:
        lines.append(f"Torch: unavailable ({exc})")
    lines.append(f"HF_HOME: {os.environ.get('HF_HOME', '')}")
    return "\n".join(lines)


def text_chunks_for_ui(text: str, chunk_mode: str) -> list[str]:
    return split_long_text(text, chunk_mode)
