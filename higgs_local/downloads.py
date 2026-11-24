"""Single-file downloads with size and SHA-256 checks, resumable through a ``.part`` file."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

import requests


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    size: int | None = None,
    sha256: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Download ``url`` to ``dest``; a wrong size or checksum is an error and the file is not kept."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "higgs-audio-easy-gui"}
    if have:
        headers["Range"] = f"bytes={have}-"
    print(f"[download] {url} -> {dest}", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 416:
            have = 0
        elif resp.status_code not in (200, 206):
            raise RuntimeError(f"Download failed: HTTP {resp.status_code} for {url}")
        mode = "ab" if resp.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        total = size or (int(resp.headers.get("Content-Length", 0)) + have)
        done = have
        last = 0.0
        with open(part, mode) as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
                done += len(block)
                now = time.time()
                if progress and now - last > 0.5:
                    last = now
                    progress(done / max(total, 1), f"{dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    got = part.stat().st_size
    if size is not None and got != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: got {got} bytes, expected {size}")
    if sha256 and sha256_of(part) != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: the checksum does not match the published one")
    os.replace(part, dest)
    if progress:
        progress(1.0, f"{dest.name} ready")
    return dest
