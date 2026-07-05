from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

from .paths import CACHE_DIR, LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, ROOT, TEMP_DIR
from .training_utils import slugify


class TrainingProcessManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.tensorboard_process: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self.started_at: float | None = None
        self.run_name = ""
        self.stop_file: Path | None = None
        self._tail: deque[str] = deque(maxlen=240)
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None

    def _progress_from_tail(self) -> tuple[int | None, int | None]:
        with self._lock:
            lines = list(self._tail)
        for line in reversed(lines):
            match = re.search(r"optimizer_step=(\d+)/(\d+)", line)
            if match:
                return int(match.group(1)), int(match.group(2))
            match = re.search(r"(\d+)\s*/\s*(\d+)\s+step", line)
            if match:
                return int(match.group(1)), int(match.group(2))
        return None, None

    def _eta_line(self, elapsed: int) -> str:
        step, total = self._progress_from_tail()
        if not step or not total or step <= 0 or total <= step:
            return ""
        seconds_per_step = elapsed / max(step, 1)
        remaining = int((total - step) * seconds_per_step)
        pct = 100.0 * step / max(total, 1)
        width = 28
        filled = int(width * min(step, total) / max(total, 1))
        bar = "█" * filled + "░" * (width - filled)
        speed = step / max(elapsed, 1)
        return (
            f"\n\nProgress\n"
            f"`|{bar}| {pct:.1f}%`\n\n"
            f"`{step}/{total}` steps · ETA `{self._format_duration(remaining)}` · "
            f"{speed:.3f} step/s"
        )

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(int(seconds), 0)
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, command: list[str], run_name: str) -> tuple[str, str]:
        if self.is_running():
            raise RuntimeError(f"Training is already running: {self.run_name}")

        safe_name = slugify(run_name, "higgs_v2_lora")
        log_dir = LOGS_DIR / "training" / safe_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.run_name = safe_name
        self.stop_file = TEMP_DIR / f"training_stop_{safe_name}.flag"
        try:
            self.stop_file.unlink(missing_ok=True)
        except Exception:
            pass
        self.started_at = time.time()
        with self._lock:
            self._tail.clear()

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HOME", str(MODELS_DIR))
        env.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
        env.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
        env.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
        env.setdefault("UV_CACHE_DIR", str(CACHE_DIR / "uv"))
        env.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR))
        env.setdefault("HIGGS_OUTPUT_DIR", str(OUTPUTS_DIR))
        env["HIGGS_TRAINING_STOP_FILE"] = str(self.stop_file)
        env["PYTHONPATH"] = str(ROOT / "train-higgs-audio") + os.pathsep + env.get("PYTHONPATH", "")

        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        print(f"[training] Starting {safe_name}", flush=True)
        print(f"[training] Log file: {self.log_path}", flush=True)
        print("[training] Command:", subprocess.list2cmdline(command), flush=True)

        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._reader_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader_thread.start()
        return self.status_markdown(), self.tail_text()

    def _append_line(self, line: str) -> None:
        clean = line.rstrip("\n")
        with self._lock:
            self._tail.append(clean)
        print(clean, flush=True)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(clean + "\n")

    def _drain_stdout(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._append_line(line)
        finally:
            code = proc.poll()
            if code is None:
                code = proc.wait()
            self._append_line(f"[training] Process exited with code {code}.")

    def stop(self) -> tuple[str, str]:
        if not self.is_running():
            return self.status_markdown(), self.tail_text() or "No Higgs training process is running."

        assert self.process is not None
        self._append_line("[training] Stop requested. Asking trainer to save and exit gracefully.")
        if self.stop_file:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop\n", encoding="utf-8")
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._append_line("[training] Graceful stop timed out; terminating process.")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._append_line("[training] Terminate timed out; killing process.")
                self.process.kill()
                self.process.wait(timeout=10)
        try:
            if self.stop_file:
                self.stop_file.unlink(missing_ok=True)
        except Exception:
            pass
        return self.status_markdown(), self.tail_text()

    def status_markdown(self) -> str:
        if self.process is None:
            return "### Training Status\nIdle."
        code = self.process.poll()
        elapsed = int(time.time() - self.started_at) if self.started_at else 0
        log_line = f"\n\nLog: `{self.log_path}`" if self.log_path else ""
        eta_line = self._eta_line(elapsed)
        if code is None:
            return (
                f"### Training Status\nRunning `{self.run_name}` for {self._format_duration(elapsed)}. "
                f"PID `{self.process.pid}`.{eta_line}{log_line}"
            )
        return (
            f"### Training Status\nFinished `{self.run_name}` with exit code `{code}` "
            f"after {self._format_duration(elapsed)}.{eta_line}{log_line}"
        )

    def tail_text(self) -> str:
        with self._lock:
            return "\n".join(self._tail)

    def refresh(self) -> tuple[str, str]:
        return self.status_markdown(), self.tail_text()

    def _archive_root_tensorboard_events(self, logdir: Path) -> None:
        event_files = list(logdir.glob("events.out.tfevents*")) if logdir.exists() else []
        if not event_files:
            return
        legacy_dir = logdir.parent / "_legacy_tensorboard_events"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for event_file in event_files:
            target = legacy_dir / event_file.name
            if target.exists():
                target = legacy_dir / f"{event_file.name}.{int(time.time())}"
            shutil.move(str(event_file), str(target))
        self._append_line(
            f"[tensorboard] Archived {len(event_files)} root event file(s) into {legacy_dir} "
            "so separate training runs do not draw as one zig-zag curve."
        )

    def launch_tensorboard(self, run_name: str | None = None) -> tuple[str, str]:
        if self.tensorboard_process is not None and self.tensorboard_process.poll() is None:
            return "### TensorBoard\nAlready running at http://127.0.0.1:6006", self.tail_text()
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:
            message = (
                "### TensorBoard\nMissing runtime dependency `setuptools`.\n\n"
                "TensorBoard imports `pkg_resources`, which is provided by setuptools. "
                "Run `uv pip install setuptools` or rerun `install.bat`."
            )
            self._append_line("[tensorboard] Missing setuptools/pkg_resources. Run: uv pip install setuptools")
            return message, self.tail_text()

        logdir = ROOT / "exp"
        if run_name:
            safe_name = slugify(run_name, "higgs_v2_lora")
            tensorboard_candidate = ROOT / "exp" / safe_name / "tensorboard"
            run_candidate = ROOT / "exp" / safe_name
            training_log_candidate = LOGS_DIR / "training" / safe_name
            if tensorboard_candidate.exists():
                logdir = tensorboard_candidate
            elif run_candidate.exists():
                logdir = run_candidate
            elif training_log_candidate.exists():
                logdir = training_log_candidate
        logdir.mkdir(parents=True, exist_ok=True)
        if logdir.name == "tensorboard":
            self._archive_root_tensorboard_events(logdir)
            subrun_dirs = [
                path
                for path in logdir.iterdir()
                if path.is_dir() and any(path.glob("events.out.tfevents*"))
            ]
            if not subrun_dirs:
                project_logdir = logdir.parent
                self._append_line(
                    f"[tensorboard] No isolated sub-run folders found in {logdir}; using project logdir {project_logdir}."
                )
                logdir = project_logdir
        url = "http://127.0.0.1:6006"
        tb_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--host", "127.0.0.1", "--port", "6006"]
        if sys.platform.startswith("win"):
            launcher = TEMP_DIR / "tensorboard_launch.cmd"
            launcher.write_text(
                "@echo off\r\n"
                "title Higgs Audio TensorBoard\r\n"
                f"cd /d {subprocess.list2cmdline([str(ROOT)])}\r\n"
                f"{subprocess.list2cmdline(tb_cmd)}\r\n",
                encoding="utf-8",
            )
            cmd = [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/k",
                str(launcher),
            ]
        else:
            cmd = tb_cmd
        print("[tensorboard] Command:", subprocess.list2cmdline(cmd), flush=True)
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        else:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.tensorboard_process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            creationflags=creationflags,
        )

        def _open_browser() -> None:
            time.sleep(2)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()
        return f"### TensorBoard\nStarted at {url}\n\nLogdir: `{logdir}`", self.tail_text()
