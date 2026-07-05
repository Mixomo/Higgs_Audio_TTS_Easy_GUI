# Higgs Audio V2/V3 Simple GUI

Local Windows WebUI for **Higgs Audio** inference, voice references, dataset preparation, and LoRA training.

This app is built around a practical local workflow: prepare reference samples, generate speech with Higgs V2 or V3, build datasets, and train LoRA adapters without sending audio to external services.

---

## What It Supports

- **Higgs V3 TTS** local Transformers inference.
- **Higgs V2 TTS** local Transformers inference.
- **LoRA loading** for V2 and V3 from `exp/`.
- **Reference voice samples** from `samples/`.
- **Faster-Whisper ASR** for reference transcription.
- **Single inference**, **dialogue**, and **long-form/batch** generation.
- **Dataset preparation** from curated audio folders.
- **Single-speaker LoRA training** for V2 and experimental V3 training.
- Local model/download/cache/output folders.

---

## Workflow

1. **Voice Library**
   - Add reference audio samples.
   - Save each sample with its transcript.
   - Samples are stored in `samples/`.

2. **Inference**
   - Select Higgs V2 or V3.
   - Optionally select a LoRA adapter.
   - Optionally select or upload reference audio.
   - Generate speech from target text.
   - Outputs can be saved to `outputs/` or kept as temporary preview files.

3. **Dialogue / Multi-Speaker**
   - Add speaker rows.
   - Assign a voice sample per speaker.
   - Generate turns independently and concatenate them with silence.

4. **Long-Form / Batch**
   - Split long text by paragraphs, periods, lines, or speaker turns.
   - Generate chunks and join the final WAV.

5. **Dataset Preparation**
   - Point the app to a folder of curated audio.
   - Existing `.txt` or `.json` transcripts are used first.
   - Missing transcripts can be generated with Faster-Whisper.
   - Train/eval splits are written under `data/`.

6. **Training**
   - Select a project and dataset.
   - Configure LoRA hyperparameters.
   - Resume from discovered checkpoints when available.
   - Training outputs go under `exp/`.

---

## Installation

Run from the project folder:

```bat
install.bat
```

The installer:

- Installs `uv` if needed.
- Creates the local virtual environment.
- Installs non-Torch dependencies.
- Installs the selected PyTorch backend last.
- Verifies CUDA when a CUDA backend is selected.

PyTorch backend choices:

1. Auto-detect NVIDIA / CPU
2. NVIDIA GTX 10xx Pascal - CUDA 11.8
3. NVIDIA RTX 20xx/30xx - CUDA 12.6
4. NVIDIA RTX 40xx/50xx - CUDA 12.8
5. CPU only
6. Windows AMD DirectML experimental

---

## Launch

```bat
start.bat
```

The app opens at:

```text
http://127.0.0.1:7860
```

---

## Requirements

### Software

- Windows 10/11.
- `uv` package manager.
- Modern browser.
- NVIDIA driver compatible with the selected CUDA backend.

### Hardware

| Task | Minimum | Recommended |
| --- | --- | --- |
| Faster-Whisper ASR | CPU or small GPU | 8 GB+ VRAM for large-v3 |
| Higgs V2 inference | 12 GB VRAM | 16 GB+ VRAM |
| Higgs V3 inference | 16 GB VRAM | 24 GB+ VRAM |
| LoRA training | 24 GB VRAM | 24 GB+ VRAM |

CPU mode is available for compatibility, but TTS inference and training are expected to be slow.

---

## Local Folders

| Folder | Purpose |
| --- | --- |
| `models/` | Final downloaded model folders |
| `models/.cache/` | uv, Hugging Face, Xet, Torch, temp, and runtime caches |
| `samples/` | Reference voice samples and transcripts |
| `outputs/` | Persisted generated WAV files |
| `data/` | Prepared training datasets |
| `exp/` | Training projects, LoRA adapters, checkpoints, eval audio |
| `logs/` | App and training logs |
| `config/` | Local UI settings |

The app does not need a root `temp/`, `cache/`, or `voices/` folder.

---

## Supported Input Formats

Reference samples and dataset audio:

```text
.wav
.mp3
.flac
.ogg
.m4a
```

Transcripts:

```text
.txt
.json
```

For samples, use the same stem:

```text
my_voice.wav
my_voice.txt
```

Generated audio is saved as:

```text
<model>_<reference_sample>_<target_text_excerpt>_<YYYYMMDD_HHMMSS_microseconds>.wav
```

The filename excerpt stops at the first natural closing mark: `.`, `...`, `…`, `?`, `!`, or `)`.

---

## Notes

- Higgs V3 can stop a single generation around its internal end signal even when `max_new_tokens` is high. Use chunking for long text.
- Every chunk is a separate synthesis pass, so small timbre/prosody variation between chunks is normal.
- Lower temperature and top-p values usually improve stability.
- V3 training is experimental and unofficial; V2 training follows the documented supervised LoRA workflow more closely.
- Torch compile is optional, CUDA-only here, and needs Triton (`install.bat` installs `triton-windows` for CUDA backends). First generation can be slower because kernels need warmup.

---

## RTX 3090 Compile Benchmark

GUI Higgs V3 test on **NVIDIA GeForce RTX 3090**, same text, seed, and frame limit.

```text
Max audio frames: 4096
Precision: auto -> bf16
Attention: SDPA
LoRA: none
Reference audio: none
```

| Mode | Load time | Frames | Audio length | Generation elapsed | RTF | Avg frames/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| torch.compile off | 8.37s | 525 | 20.7s | 21s | 1.01 | 24.95 |
| torch.compile on | 8.32s | 551 | 21.8s | 9s | 0.41 | 57.98 |

In this run, `torch.compile` was about **2.3x faster** during frame generation once active. First-time compilation can still pause on new shapes.

---

## Clean Install Test

To simulate a fresh user environment without deleting models or outputs:

```powershell
Remove-Item -LiteralPath ".venv" -Recurse -Force
Remove-Item -LiteralPath "models\.cache\uv" -Recurse -Force
```

Then run:

```bat
install.bat
start.bat
```

---

## Credits

Built for local use around:

- [Boson AI Higgs Audio](https://github.com/boson-ai/higgs-audio)
- [train-higgs-audio](https://github.com/boson-ai/higgs-audio)
- Faster-Whisper / CTranslate2
- Gradio
