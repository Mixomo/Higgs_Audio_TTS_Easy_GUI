"""Download audio from YouTube (or any site yt-dlp supports) as dataset input.

Each URL may be a video, a playlist or a channel tab. Audio is extracted to
WAV with the ffmpeg binary bundled by ``imageio-ffmpeg``; YouTube's player
JavaScript runs on the bundled ``deno`` runtime (no system installs needed). Optional ``start-end`` sections cut only part of a video, e.g.

    https://youtu.be/abc123
    https://youtu.be/def456 1:20-5:00
    https://www.youtube.com/playlist?list=PL...

Nothing but the audio is written: no subtitles or info JSON, because the
dataset builder would treat a sidecar ``.txt`` / ``.json`` as a transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import ROOT

DOWNLOADS_DIR = ROOT / "downloads"
_TIME = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


@dataclass
class DownloadResult:
    folder: Path
    files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _seconds(value: str) -> float:
    m = _TIME.match(value.strip())
    if not m:
        raise ValueError(f"Bad time '{value}' (use 90, 1:30 or 1:02:30)")
    parts = [p for p in m.groups() if p is not None]
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def parse_lines(text: str) -> list[tuple[str, tuple[float, float] | None]]:
    """[(url, (start, end) or None)] from one URL per line (optionally 'URL start-end')."""
    out = []
    for raw in (text or "").replace(",", "\n").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        url, section = parts[0], None
        if len(parts) > 1 and "-" in parts[1]:
            a, b = parts[1].split("-", 1)
            start, end = _seconds(a), _seconds(b)
            if end <= start:
                raise ValueError(f"Section end must be after its start: {line}")
            section = (start, end)
        if not re.match(r"^https?://", url):
            raise ValueError(f"Not a URL: {url}")
        out.append((url, section))
    return out


def download_audio(
    urls_text: str,
    folder: Path,
    progress: Callable[[float, str], None] | None = None,
) -> DownloadResult:
    """Download the audio of every URL into ``folder`` as WAV. Already downloaded videos are skipped."""
    import imageio_ffmpeg
    import yt_dlp
    from yt_dlp.utils import download_range_func

    items = parse_lines(urls_text)
    if not items:
        raise ValueError("Paste at least one YouTube URL (one per line).")
    folder.mkdir(parents=True, exist_ok=True)
    result = DownloadResult(folder=folder)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    js_runtimes = {}
    try:  # YouTube needs a JavaScript runtime; the app bundles deno
        import deno

        js_runtimes = {"deno": {"path": deno.find_deno_bin()}}
    except Exception:
        pass

    for n, (url, section) in enumerate(items):
        base = n / len(items)

        def hook(d, base=base):
            if not progress:
                return
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                frac = (d.get("downloaded_bytes") or 0) / total if total else 0.0
                name = Path(d.get("filename") or "").stem[:60]
                progress(base + frac / len(items), f"Downloading {name} ({frac:.0%})")
            elif d.get("status") == "finished":
                progress(base + 1 / len(items), "Extracting audio")

        suffix = f" [{int(section[0])}-{int(section[1])}s]" if section else ""
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(folder / f"%(title).80B [%(id)s]{suffix}.%(ext)s"),
            "ffmpeg_location": ffmpeg,
            "js_runtimes": js_runtimes,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", "44100"]},
            "download_archive": str(folder / ".downloaded.txt") if not section else None,
            "ignoreerrors": True,
            "noplaylist": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "writeinfojson": False,
            "writethumbnail": False,
            "windowsfilenames": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "progress_hooks": [hook],
        }
        if section:
            opts["download_ranges"] = download_range_func(None, [section])
            opts["force_keyframes_at_cuts"] = True
        before = set(folder.glob("*.wav"))
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                code = ydl.download([url])
            if code:
                result.errors.append(f"{url}: some items could not be downloaded (private, removed or region-locked)")
        except Exception as exc:  # noqa: BLE001 - report and continue with the next URL
            result.errors.append(f"{url}: {exc}")
        result.files.extend(sorted(set(folder.glob("*.wav")) - before))
    if progress:
        progress(1.0, "Download finished")
    return result


def describe_folder(folder: Path) -> str:
    wavs = sorted(folder.glob("*.wav")) if folder.exists() else []
    if not wavs:
        return "No audio downloaded yet."
    import soundfile as sf

    total = 0.0
    for w in wavs:
        try:
            total += sf.info(str(w)).duration
        except Exception:
            pass
    names = "\n".join(f"- {w.name}" for w in wavs[:12]) + ("\n- ..." if len(wavs) > 12 else "")
    return f"**{len(wavs)} audio file(s)**, {total / 60:.1f} min in `{folder}`\n\n{names}"


# ---------------------------------------------------------------- one reference clip (Inference, Voice Library)
REFERENCES_DIR = DOWNLOADS_DIR / "references"
_URL_T = re.compile(r"[?&#](?:t|start)=(?:(\d+)h)?(?:(\d+)m)?(\d+)s?(?:&|$)")


def url_start(url: str) -> float | None:
    """The start time a shared link carries (``?t=90``, ``&t=1m30s``), or None."""
    m = _URL_T.search(url or "")
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return float(h * 3600 + mi * 60 + s)


def clip_times(url: str, start: str = "", end: str = "") -> tuple[float, float | None]:
    """(start, end) in seconds. Empty start: the link's ``t=`` or 0; empty end: None, up to the end of the video
    (nothing is cut to a fixed length: the whole part asked for is downloaded)."""
    s = _seconds(start) if (start or "").strip() else (url_start(url) or 0.0)
    e = _seconds(end) if (end or "").strip() else None
    if e is not None and e <= s:
        raise ValueError("“To” must be after “From”.")
    return s, e


def download_clip(
    url: str,
    start: str = "",
    end: str = "",
    folder: Path = REFERENCES_DIR,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, str]:
    """Download one video's audio (all of it, or the From / To part) as a mono WAV. Returns (file, video title)."""
    import imageio_ffmpeg
    import yt_dlp
    from yt_dlp.utils import download_range_func

    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        raise ValueError("Paste a video link (https://…).")
    s, e = clip_times(url, start, end)
    folder.mkdir(parents=True, exist_ok=True)
    js_runtimes = {}
    try:  # YouTube needs a JavaScript runtime; the app bundles deno
        import deno

        js_runtimes = {"deno": {"path": deno.find_deno_bin()}}
    except Exception:
        pass

    def hook(d):
        if not progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes") or 0) / total if total else 0.0
            progress(0.1 + 0.8 * frac, f"Downloading ({frac:.0%})")
        elif d.get("status") == "finished":
            progress(0.92, "Extracting audio")

    whole = s <= 0 and e is None
    part = "" if whole else f" {int(s)}-{int(e)}s" if e is not None else f" {int(s)}s-end"
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(folder / f"%(title).60B [%(id)s]{part}.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "js_runtimes": js_runtimes,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", "44100"]},
        "noplaylist": True,  # a link inside a playlist means that one video
        "overwrites": True,
        "windowsfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    if not whole:  # only that part is downloaded
        opts["download_ranges"] = download_range_func(None, [(s, e if e is not None else float("inf"))])
        opts["force_keyframes_at_cuts"] = True
    if progress:
        progress(0.02, "Reading the link")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist":  # a playlist link: its first video
            info = next((x for x in info.get("entries") or [] if x), None) or {}
        done = info.get("requested_downloads") or []
        path = Path(done[0]["filepath"]) if done and done[0].get("filepath") else Path(ydl.prepare_filename(info)).with_suffix(".wav")
    if not path.is_file():
        raise RuntimeError("Nothing was downloaded (private, removed or region-locked video?).")
    if progress:
        progress(1.0, "Download finished")
    return path, str(info.get("title") or path.stem)
