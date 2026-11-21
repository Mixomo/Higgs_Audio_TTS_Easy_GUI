"""Keep inference at full speed on Windows when the console is not focused.

Windows 11 derives a process' scheduling QoS from the state of its window. A
console app (python under cmd/conhost) that is not the foreground window, or
is covered by the browser, drops from High QoS to Medium/Low/Eco QoS:

* its threads can be moved to efficiency cores and run at lower clocks
  (execution-speed throttling / EcoQoS), and
* its timer-resolution requests are ignored, so short waits round up to the
  default 15.6 ms tick.

The Higgs decode loop launches many tiny CUDA kernels per frame and syncs with
the GPU every frame, so it is CPU/latency bound and both effects slash the
frames/s. Explicitly opting out of these policies restores High QoS.
"""

from __future__ import annotations

import sys
import threading

_IS_WINDOWS = sys.platform.startswith("win")

# PROCESS_POWER_THROTTLING_* / THREAD_POWER_THROTTLING_* control masks.
_EXECUTION_SPEED = 0x1
_IGNORE_TIMER_RESOLUTION = 0x4  # process-only, Windows 11+

_ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS
_ThreadPowerThrottling = 3  # THREAD_INFORMATION_CLASS

_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000

_tuned_threads: set[int] = set()
_tuned_lock = threading.Lock()


def _power_state(control_mask: int):
    import ctypes

    class POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32), ("StateMask", ctypes.c_uint32)]

    # StateMask=0 with the bit set in ControlMask means "never throttle this".
    return POWER_THROTTLING_STATE(Version=1, ControlMask=control_mask, StateMask=0)


def configure_process() -> str:
    """Disable background throttling for the whole process. Returns a status line."""
    if not _IS_WINDOWS:
        return ""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetProcessInformation.restype = ctypes.c_int
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()

    throttling = "unavailable"
    for mask, label in (
        (_EXECUTION_SPEED | _IGNORE_TIMER_RESOLUTION, "disabled (speed+timer)"),
        (_EXECUTION_SPEED, "disabled (speed)"),  # Windows 10: timer flag unsupported
    ):
        state = _power_state(mask)
        if kernel32.SetProcessInformation(process, _ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state)):
            throttling = label
            break

    # Request 1 ms timer resolution (honoured even in background thanks to the
    # IGNORE_TIMER_RESOLUTION opt-out above).
    timer = "default"
    try:
        winmm = ctypes.windll.winmm
        if winmm.timeBeginPeriod(1) == 0:
            timer = "1ms"
            import atexit

            atexit.register(winmm.timeEndPeriod, 1)
    except Exception:
        pass

    priority_ok = bool(kernel32.SetPriorityClass(process, _ABOVE_NORMAL_PRIORITY_CLASS))
    configure_current_thread()
    quick_edit = disable_console_quick_edit()
    return (
        f"[runtime] Windows background-throttling={throttling}; timer={timer}; "
        f"process-priority={'above-normal' if priority_ok else 'unchanged'}; quick-edit={quick_edit}."
    )


def disable_console_quick_edit() -> str:
    """Turn off the console's QuickEdit mode.

    With QuickEdit on, a single click inside the cmd window starts a text
    selection and Windows blocks every write to the console until the selection
    is cleared. Since the generation loop prints progress, inference freezes
    until a key is pressed. Set HIGGS_KEEP_QUICKEDIT=1 to keep it.
    """
    if not _IS_WINDOWS:
        return "n/a"
    import os

    if os.environ.get("HIGGS_KEEP_QUICKEDIT", "").strip() in {"1", "true", "yes"}:
        return "kept"
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080  # required for the QuickEdit bit to take effect
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        if not handle or not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return "no-console"
        new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        if kernel32.SetConsoleMode(handle, new_mode):
            return "off"
        return "unchanged"
    except Exception:
        return "unchanged"


def configure_current_thread() -> None:
    """Mark the calling thread as High QoS (cheap; runs once per thread).

    Gradio runs handlers in worker threads; tagging the thread that drives the
    GPU loop keeps Windows from parking it on efficiency cores.
    """
    if not _IS_WINDOWS:
        return
    ident = threading.get_ident()
    with _tuned_lock:
        if ident in _tuned_threads:
            return
        _tuned_threads.add(ident)
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetThreadInformation.restype = ctypes.c_int
        state = _power_state(_EXECUTION_SPEED)
        kernel32.SetThreadInformation(kernel32.GetCurrentThread(), _ThreadPowerThrottling, ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass
