"""The script LLM: a local llama.cpp server or any OpenAI-compatible server (LM Studio, Ollama, llama-server).

Local mode runs ``tools/llama/llama-server.exe`` on a GGUF from ``models/llm`` (the
same Gemma 4 12B builds Speech Studio recommends), started on first use and
stopped by "Unload LLM" or before voicing when asked to free VRAM. The director
operations (podcast script, continue, tags, punctuation, review, direct speech,
accents) ask for JSON by schema, and an answer that must keep the words is
checked to keep them, as Speech Studio's director does.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import threading
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from .downloads import download_file
from .paths import CACHE_DIR, LOGS_DIR, MODELS_DIR, ROOT

LLAMA_DIR = ROOT / "tools" / "llama"
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
LLM_DIR = MODELS_DIR / "llm"
# llama.cpp's Windows CUDA 13 build and its CUDA runtime, from one GitHub release (sizes and sha256 as GitHub lists them)
LLAMA_ZIPS = [
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11900/llama-b11900-bin-win-cuda-13.4-x64.zip",
     164_721_212, "0d75ad36367da00622c146a97e82fac3dcfcbe960afc2f0234eef026886b16f6"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11900/cudart-llama-bin-win-cuda-13.4-x64.zip",
     423_535_356, "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668"),
]


@dataclass(frozen=True)
class LocalModel:
    label: str
    file: str
    url: str
    size: int
    sha256: str


# Speech Studio's recommended script models (Gemma 4 12B instruct), pinned to their commits.
LOCAL_MODELS = [
    LocalModel("Gemma 4 12B · QAT Q4_0 · 7.0 GB · ~9 GB VRAM (recommended next to Higgs)", "gemma-4-12b-it-qat-q4_0.gguf",
               "https://huggingface.co/google/gemma-4-12b-it-qat-q4_0-gguf/resolve/2b318d6ebebf093f50ca4376e858325f10703358/gemma-4-12b-it-qat-q4_0.gguf",
               6_975_877_728, "faff1a63667fac17ac5e777f47114688fcefea96e220e211aaa8d62c2c4561f1"),
    LocalModel("Gemma 4 12B · Q5_K_M · 8.4 GB · ~10 GB VRAM", "gemma-4-12b-it-Q5_K_M.gguf",
               "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/d997c805aafe035a8024f961c6e1afd6b30d79a5/gemma-4-12b-it-Q5_K_M.gguf",
               8_413_574_560, "1bc633ec98817858bec10f73fa026481c9662449aae4b80a05dfb28ef784c278"),
    LocalModel("Gemma 4 12B · Q6_K · 9.8 GB · ~12 GB VRAM", "gemma-4-12b-it-Q6_K.gguf",
               "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/d997c805aafe035a8024f961c6e1afd6b30d79a5/gemma-4-12b-it-Q6_K.gguf",
               9_786_021_280, "e1602ddc224c159584eb4c7d6a6c8d682fc6afb2efb8f76c10bfd63ba71436a2"),
    LocalModel("Gemma 4 12B · Q8_0 · 12.7 GB · ~15 GB VRAM (best)", "gemma-4-12b-it-Q8_0.gguf",
               "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/d997c805aafe035a8024f961c6e1afd6b30d79a5/gemma-4-12b-it-Q8_0.gguf",
               12_669_646_240, "74d2d4f0b5b08ca8589d1a5f50e689c0984469f3cedbdc7d67458c6e9e35496a"),
]
LOCAL_BY_LABEL = {m.label: m for m in LOCAL_MODELS}
SETTINGS_FILE = ROOT / "config" / "llm_settings.json"


def model_path(m: LocalModel) -> Path:
    return LLM_DIR / m.file


def model_installed(m: LocalModel) -> bool:
    p = model_path(m)
    return p.is_file() and p.stat().st_size == m.size


def other_ggufs() -> list[Path]:
    """GGUFs the person put in models/llm themselves."""
    known = {m.file for m in LOCAL_MODELS}
    return sorted(p for p in LLM_DIR.glob("*.gguf") if p.name not in known and "mmproj" not in p.name.lower()) if LLM_DIR.is_dir() else []


def local_choices() -> list[tuple[str, str]]:
    out = []
    for m in LOCAL_MODELS:
        out.append((("✓ " if model_installed(m) else "⬇ ") + m.label, m.label))
    for p in other_ggufs():
        out.append((f"✓ {p.name} (your file)", str(p)))
    return out


def default_local_choice() -> str:
    for m in reversed(LOCAL_MODELS):
        if model_installed(m):
            return m.label
    return LOCAL_MODELS[0].label


def llama_installed() -> bool:
    return LLAMA_SERVER.is_file() and (LLAMA_DIR / "ggml-cuda.dll").is_file() and (LLAMA_DIR / "cudart64_13.dll").is_file()


def ensure_llama(progress=None) -> Path:
    if llama_installed():
        return LLAMA_SERVER
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    for url, size, sha in LLAMA_ZIPS:
        zpath = CACHE_DIR / "downloads" / url.rsplit("/", 1)[1]
        download_file(url, zpath, size, sha, progress=progress)
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if info.is_dir() or not name:
                    continue
                with zf.open(info) as src, open(LLAMA_DIR / name, "wb") as dst:
                    dst.write(src.read())
        zpath.unlink(missing_ok=True)
    if not llama_installed():
        raise RuntimeError(f"llama.cpp was unpacked but {LLAMA_SERVER} or its CUDA runtime is missing")
    return LLAMA_SERVER


def ensure_local_model(choice: str, progress=None) -> Path:
    m = LOCAL_BY_LABEL.get(choice)
    if m is None:
        p = Path(choice)
        if p.is_file():
            return p
        raise FileNotFoundError(f"LLM model not found: {choice}")
    if model_installed(m):
        return model_path(m)
    return download_file(m.url, model_path(m), m.size, m.sha256, progress=progress)


# ---------------------------------------------------------------- the local server
class LocalServer:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.port: int | None = None
        self.lock = threading.Lock()
        self.log_path = LOGS_DIR / "llama-server.log"

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, model_file: Path, ctx: int = 16384, gpu_layers: int = 99, progress=None) -> str:
        with self.lock:
            if self.running and self.model == str(model_file):
                return self.base_url()
            self.stop_locked()
            ensure_llama(progress)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "w", encoding="utf-8", errors="replace")
            cmd = [str(LLAMA_SERVER), "-m", str(model_file), "--host", "127.0.0.1", "--port", str(self.port),
                   "-c", str(int(ctx)), "-ngl", str(int(gpu_layers)), "--jinja", "-fa", "on", "--no-webui",
                   "--reasoning", "off", "-np", "1"]
            print(f"[llm] starting llama-server: {model_file.name} (ctx {ctx}, gpu layers {gpu_layers})", flush=True)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(cmd, cwd=str(LLAMA_DIR), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            self.model = str(model_file)
            t0 = time.time()
            while time.time() - t0 < 600:
                if self.proc.poll() is not None:
                    tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                    self.proc = None
                    raise RuntimeError(f"llama-server stopped while loading the model:\n{tail}")
                try:
                    r = requests.get(self.base_url() + "/health", timeout=2)
                    if r.status_code == 200:
                        print(f"[llm] llama-server ready in {time.time() - t0:.0f} s on port {self.port}", flush=True)
                        return self.base_url()
                except requests.RequestException:
                    pass
                if progress:
                    progress(None, f"Loading {model_file.name}… {time.time() - t0:.0f} s")
                time.sleep(0.5)
            self.stop_locked()
            raise TimeoutError("llama-server did not become ready in 10 minutes")

    def stop_locked(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            print("[llm] llama-server stopped (VRAM freed)", flush=True)
        self.proc = None
        self.model = None

    def stop(self) -> str:
        with self.lock:
            was = self.running
            self.stop_locked()
        return "LLM unloaded: llama-server stopped." if was else "The local LLM was not loaded."


SERVER = LocalServer()


# ---------------------------------------------------------------- the client
@dataclass
class LLMConfig:
    provider: str = "Local (llama.cpp)"
    local_model: str = ""
    url: str = "http://127.0.0.1:1234/v1"
    api_key: str = ""
    remote_model: str = ""
    ctx: int = 16384
    gpu_layers: int = 99
    temperature: float = 0.7
    free_tts: bool = True

    @property
    def local(self) -> bool:
        return self.provider.startswith("Local")


def _v1(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Type the server address, e.g. http://127.0.0.1:1234/v1 for LM Studio.")
    return url if url.endswith("/v1") else url + "/v1"


def list_remote_models(url: str, api_key: str = "") -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(_v1(url) + "/models", headers=headers, timeout=8)
    r.raise_for_status()
    data = r.json().get("data", [])
    return [m.get("id") for m in data if m.get("id")]


def check_remote(url: str, api_key: str = "") -> tuple[bool, str, list[str]]:
    try:
        models = list_remote_models(url, api_key)
    except requests.ConnectionError:
        return False, f"No server answers at {url}. Is LM Studio's (or the other) server started?", []
    except requests.HTTPError as exc:
        return False, f"The server answered {exc.response.status_code}: {exc.response.text[:200]}", []
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", []
    if not models:
        return True, f"Connected to {url}, but it lists no models: load one in the server.", []
    return True, f"Connected to {url}: {len(models)} model(s) — {', '.join(models[:6])}{'…' if len(models) > 6 else ''}", models


class LLMError(RuntimeError):
    pass


def _endpoint(cfg: LLMConfig, progress=None, before_local=None) -> tuple[str, str, dict]:
    if cfg.local:
        if not SERVER.running and before_local:
            before_local()
        model = ensure_local_model(cfg.local_model or default_local_choice(), progress)
        base = SERVER.start(model, cfg.ctx, cfg.gpu_layers, progress)
        return base + "/v1", "local", {}
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    model = cfg.remote_model
    if not model:
        models = list_remote_models(cfg.url, cfg.api_key)
        if not models:
            raise LLMError("The server lists no models: load one first.")
        model = models[0]
    return _v1(cfg.url), model, headers


def chat(cfg: LLMConfig, system: str, user: str, schema: dict | None = None, max_tokens: int = 4096,
         temperature: float | None = None, progress=None, before_local=None) -> str:
    base, model, headers = _endpoint(cfg, progress, before_local)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": float(cfg.temperature if temperature is None else temperature),
        "top_p": 0.9,
        "max_tokens": int(max_tokens),
        "stream": False,
        # scripts need no thinking: a reasoning model would spend the token budget on it (Gemma 4, Qwen 3)
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none" if not cfg.local else None,
    }
    body = {k: v for k, v in body.items() if v is not None}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}
    if progress:
        progress(None, "The LLM is writing…")
    t0 = time.time()
    r = requests.post(base + "/chat/completions", json=body, headers=headers, timeout=900)
    if r.status_code == 400 and schema is not None:
        # a server that does not take a JSON schema: ask again for plain JSON
        body.pop("response_format", None)
        r = requests.post(base + "/chat/completions", json=body, headers=headers, timeout=900)
    if r.status_code != 200:
        raise LLMError(f"The LLM server answered {r.status_code}: {r.text[:300]}")
    msg = r.json()["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip() and (msg.get("reasoning_content") or "").strip():
        # a server that still thought: the answer may be at the end of the thoughts
        text = msg["reasoning_content"]
    print(f"[llm] answer in {time.time() - t0:.1f} s ({len(text)} chars)", flush=True)
    return text


def ask_json(cfg: LLMConfig, system: str, user: str, schema: dict, max_tokens: int = 4096, temperature=None,
             progress=None, before_local=None) -> dict:
    text = chat(cfg, system, user, schema, max_tokens, temperature, progress, before_local).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise LLMError(f"The LLM did not answer JSON: {text[:200]}")
    try:
        return json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        try:
            import json_repair

            return json_repair.loads(text[a : b + 1])
        except Exception as exc:
            raise LLMError(f"The LLM's JSON could not be read ({exc}): {text[:200]}") from exc


# ---------------------------------------------------------------- the director
CHUNK_CHARS = 2500
# spoken speed of the speech engine with its pauses (measured on Higgs V3: ~150 words a minute)
WORDS_PER_MINUTE = 150


def lines_schema(names: list[str] | None) -> dict:
    speaker = {"type": "string", "enum": names} if names else {"type": "string"}
    return {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": {
            "type": "object",
            "properties": {"speaker": speaker, "text": {"type": "string"}, "direction": {"type": "string"}},
            "required": ["speaker", "text", "direction"], "additionalProperties": False}}},
        "required": ["lines"], "additionalProperties": False,
    }


CLOSING_NOTES = {
    False: " This is only part of the episode and the conversation goes on after these lines: do not wrap up, sum up "
           "or say goodbye.",
    True: " Bring the episode to a natural end: the last lines wrap up and say goodbye.",
}


def podcast_rules(cast: str, lang: str, words: int, tags_note: str, closing: bool | None = None) -> str:
    return (
        f"You are a podcast scriptwriter. Write a lively spoken dialogue for these speakers: {cast}. Each speaker keeps "
        f"their own manner and character. Write in the language \"{lang}\" (\"auto\" means the language of the topic). "
        f"About {words} words in total. Each line is what one speaker says aloud: no stage directions or sound effects "
        f"inside the text, numbers as they are said. \"direction\" is a short note on how the line sounds (tone, pace), "
        f"or empty.{tags_note}{CLOSING_NOTES.get(closing, '')} Answer with JSON only."
    )


def tags_note(tag_rules: str | None) -> str:
    if not tag_rules:
        return ""
    return (" You may start a line with expression tags of the speech engine where the meaning asks for one, written "
            f"exactly as listed; use few: {tag_rules}")


def write_podcast(cfg: LLMConfig, topic: str, speakers: list[dict], lang: str, minutes: float, tag_rules: str | None = None,
                  progress=None, before_local=None, closing: bool | None = None) -> list[dict]:
    names = [s["name"] for s in speakers]
    cast = "; ".join(f"{s['name']} — {s['manner']}" if s.get("manner") else s["name"] for s in speakers)
    words = int(max(0.5, min(float(minutes), 30)) * WORDS_PER_MINUTE)
    system = podcast_rules(cast, lang, words, tags_note(tag_rules), closing)
    data = ask_json(cfg, system, topic, lines_schema(names), max(1024, min(words * 4, 16000)), 0.7, progress, before_local)
    lines = [ln for ln in data.get("lines", []) if (ln.get("text") or "").strip()]
    if not lines:
        raise LLMError("The LLM wrote no lines.")
    return lines


def script_tail(written: list[tuple[str, str]]) -> str:
    limit = CHUNK_CHARS * 2
    size = 0
    tail: list[str] = []
    for speaker, text in reversed(written):
        if size + len(text) > limit:
            if not tail:
                tail.append(f"{speaker}: …{text[-limit:]}")
            break
        size += len(text)
        tail.append(f"{speaker}: {text}")
    return "\n".join(reversed(tail))


def continue_script(cfg: LLMConfig, speakers: list[dict], written: list[tuple[str, str]], lang: str, minutes: float,
                    direction: str = "", tag_rules: str | None = None, progress=None, before_local=None,
                    closing: bool | None = None) -> list[dict]:
    names = [s["name"] for s in speakers]
    words = int(max(0.5, min(float(minutes), 30)) * WORDS_PER_MINUTE)
    system = (
        podcast_rules("; ".join(names), lang, words, tags_note(tag_rules), closing)
        + " The script so far is given, its last lines: write only what comes next, starting from the line after its last "
        "one. Do not repeat, rewrite or sum up the lines given; keep each speaker's manner and the language they speak."
    )
    wish = f"\n\nWhat comes next: {direction.strip()}" if direction.strip() else ""
    data = ask_json(cfg, system, script_tail(written) + wish, lines_schema(names), max(1024, min(words * 4, 16000)), 0.7,
                    progress, before_local)
    lines = [ln for ln in data.get("lines", []) if (ln.get("text") or "").strip()]
    if not lines:
        raise LLMError("The LLM wrote no lines.")
    return lines


TAG_LIKE = re.compile(r"<\|[^|>]*\|>|\[[^\]]*\]|</?SE[^>]*>")


def letters(text: str) -> str:
    s = unicodedata.normalize("NFKC", TAG_LIKE.sub(" ", text or "")).lower()
    return "".join(c for c in s if c.isalnum())


def letters_no_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", letters(text).replace("ё", "е"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _lang_note(lang: str) -> str:
    return "" if not lang or lang == "auto" else f" (language: {lang})"


EDIT_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
    "required": ["id", "text"], "additionalProperties": False}}}, "required": ["lines"], "additionalProperties": False}


def rewrite(cfg: LLMConfig, system: str, lines: list[dict], temperature: float, accept, progress=None, before_local=None
            ) -> tuple[list[str], int]:
    """``lines`` = [{"id", "speaker", "text"}]; the answer for each line passes ``accept(before, after)`` or keeps the line."""
    out = [ln["text"] for ln in lines]
    pieces: list[list[int]] = [[]]
    size = 0
    for i, ln in enumerate(lines):
        if size and size + len(ln["text"]) > CHUNK_CHARS:
            pieces.append([])
            size = 0
        size += len(ln["text"])
        pieces[-1].append(i)
    refused = 0
    for k, piece in enumerate(p for p in pieces if p):
        if progress:
            progress(k / max(len(pieces), 1), f"The LLM is editing part {k + 1}/{len(pieces)}…")
        user = json.dumps([{"id": str(i), "speaker": lines[i].get("speaker", ""), "text": lines[i]["text"]} for i in piece],
                          ensure_ascii=False)
        chars = sum(len(lines[i]["text"]) for i in piece)
        data = ask_json(cfg, system, user, EDIT_SCHEMA, max(1024, min(chars * 3, 16000)), temperature, None, before_local)
        for item in data.get("lines", []):
            try:
                i = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if i not in piece:
                continue
            after = accept(lines[i]["text"], item.get("text") or "")
            if after is None:
                refused += 1
            else:
                out[i] = after
    return out, refused


def add_tags(cfg, lines, lang, tag_rules: str, known_tags: set[str], progress=None, before_local=None):
    system = (
        f"You are a voice director preparing lines for the Higgs Audio speech engine{_lang_note(lang)}. Add expression "
        "tags where the meaning of a line asks for them: an emotion, a manner of speaking, a pace, a sound like a laugh or "
        f"a sigh. Use only these tags, written exactly as listed:\n{tag_rules}\n\nSentence tags go at the start of the "
        "sentence they color; a sound tag goes inline followed by its sound word. Do not change, add, remove or reorder "
        "any word or punctuation mark: only insert tags. Keep tags already in a line. Use few tags: most lines need none "
        'or one. Answer JSON {"lines": [{"id": ..., "text": ...}]} with every line given, in the same order.'
    )

    def accept(before, after):
        after = TAG_LIKE.sub(lambda m: m.group(0) if m.group(0) in known_tags or m.group(0) in before else "", after)
        return after if letters(after) == letters(before) else None

    return rewrite(cfg, system, lines, 0.4, accept, progress, before_local)


def punctuate(cfg, lines, lang, progress=None, before_local=None):
    system = (
        f"You are an editor preparing lines to be read aloud by a speech engine{_lang_note(lang)}. Fix the punctuation so "
        "each line reads naturally: commas, dashes and full stops where speech pauses, a question mark where it asks, a "
        "run-on sentence cut into two with a full stop. Do not change, add, remove or reorder any word, keep its letters "
        "and case, and keep every tag such as <|...|>, [...] or <...> exactly where it stands. Change nothing in a line "
        'that already reads well. Answer JSON {"lines": [{"id": ..., "text": ...}]} with every line given, in the same order.'
    )
    tags = lambda t: TAG_LIKE.findall(t)  # noqa: E731
    return rewrite(cfg, system, lines, 0.2, lambda b, a: a if letters(a) == letters(b) and tags(a) == tags(b) else None,
                   progress, before_local)


def accents(cfg, lines, lang, progress=None, before_local=None):
    system = (
        f"You are a proofreader preparing lines to be read aloud by a speech engine{_lang_note(lang)}. Put back every "
        "written accent the language needs (Spanish and Portuguese tildes, French accents, Russian ё where it is said ё, "
        "a stress mark on an ambiguous Russian word), so the engine pronounces each word right. Do not change, add, remove "
        "or reorder any word or punctuation, and keep every tag such as <|...|> exactly where it stands. Answer JSON "
        '{"lines": [{"id": ..., "text": ...}]} with every line given, in the same order.'
    )
    tags = lambda t: TAG_LIKE.findall(t)  # noqa: E731
    return rewrite(cfg, system, lines, 0.1,
                   lambda b, a: a if letters_no_accents(a) == letters_no_accents(b) and tags(a) == tags(b) else None,
                   progress, before_local)


def review(cfg, lines, lang, instruction: str = "", progress=None, before_local=None):
    wish = instruction.strip() or (
        "fix spelling, grammar and wording mistakes, awkward phrasing and anything that would sound wrong read aloud, "
        "keeping the meaning, the tone and roughly the length"
    )
    system = (
        f"You are a script editor reviewing lines that a speech engine reads aloud{_lang_note(lang)}. Revise each line: "
        f"{wish}. Each line stays what its speaker says aloud, in that speaker's manner and in the line's own language: no "
        "stage directions inside the text, numbers as they are said, no line merged with another or split in two. Keep "
        "every tag such as <|...|> where it belongs. A line that needs nothing stays exactly as it is. Answer JSON "
        '{"lines": [{"id": ..., "text": ...}]} with every line given, in the same order.'
    )
    return rewrite(cfg, system, lines, 0.5, lambda b, a: a.strip() or None, progress, before_local)


SPLIT_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
    "required": ["speaker", "text"], "additionalProperties": False}}}, "required": ["lines"], "additionalProperties": False}


def split_direct_speech(cfg, text: str, narrator: str, lang: str, progress=None, before_local=None) -> list[dict]:
    """A narrated text split into the narrator's lines and each character's quoted speech, word for word."""
    system = (
        f"You split a narrated text{_lang_note(lang)} into the lines of its voices for an audiobook. The narrator is "
        f"\"{narrator}\"; each character who speaks in direct speech gets a short constant name (the same name every time "
        "they speak). Give the text in order as lines {speaker, text}: narration to the narrator, every quoted or dashed "
        "speech to its character. Keep every word exactly as written, in order; drop only the quotation marks or the dash "
        "that opens a line of speech. Answer with JSON only."
    )
    out: list[dict] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunk = ""
    chunks = []
    for p in paragraphs:
        if chunk and len(chunk) + len(p) > CHUNK_CHARS:
            chunks.append(chunk)
            chunk = ""
        chunk = f"{chunk}\n\n{p}" if chunk else p
    if chunk:
        chunks.append(chunk)
    for k, c in enumerate(chunks):
        if progress:
            progress(k / max(len(chunks), 1), f"Splitting part {k + 1}/{len(chunks)}…")
        data = ask_json(cfg, system, c, SPLIT_SCHEMA, max(1024, min(len(c) * 3, 16000)), 0.2, None, before_local)
        lines = [ln for ln in data.get("lines", []) if (ln.get("text") or "").strip()]
        quotes = re.compile(r"[\"«»“”„—–-]")
        if letters(quotes.sub("", "".join(ln["text"] for ln in lines))) != letters(quotes.sub("", c)):
            # the model changed words: the part stays with the narrator as written
            lines = [{"speaker": narrator, "text": c.strip()}]
        out.extend(lines)
    return out
