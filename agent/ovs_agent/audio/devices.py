import re
import sys
import time

import pyaudio

# Names that report input channels on Jetson/desktop but are NOT real mics:
# HDMI/HDA capture stubs, the Jetson APE virtual nodes (16 phantom channels
# each), digital passthrough, and the ALSA aggregate/plug devices. The mic
# auto-detect must never fall back onto one of these — opening it yields
# PortAudio -9998 (e.g. HDMI has 0 usable channels) or silent garbage.
_PHANTOM_INPUT_RE = re.compile(
    r"(?i)\bhdmi\b|\bhda\b|jetson.*\bape\b|\bape\b|spdif|sysdefault|dmix|"
    r"surround\d|samplerate|speexrate|upmix|vdownmix|^default$|^null$"
)


def _is_real_mic(name: str) -> bool:
    """A device name that looks like an actual capture mic, not a phantom
    HDMI/HDA/APE node."""
    return not _PHANTOM_INPUT_RE.search(name or "")


def _best_fallback_input(
    devices: list[tuple[int, str]],
) -> tuple[int, str] | None:
    """Pick the most mic-like input when the reSpeaker name match misses.
    Prefer a USB-audio device (the reSpeaker family is USB; this catches it
    even if the name regex flaked during a hotplug/recreate race), then any
    non-phantom input. Never returns an HDMI/HDA/APE node."""
    real = [(i, n) for (i, n) in devices if _is_real_mic(n)]
    for i, n in real:
        if "usb" in n.lower():
            return i, n
    if real:
        return real[0]
    return None


def _log(msg: str) -> None:
    """Emit on stderr so callers that capture stdout (entrypoint MIC_INDEX
    resolution) don't pick up these informational lines as the resolved
    index value."""
    print(msg, file=sys.stderr)


def _enumerate_inputs() -> list[tuple[int, str]]:
    """Return [(index, name), ...] for all PyAudio devices with input channels."""
    audio = pyaudio.PyAudio()
    try:
        out: list[tuple[int, str]] = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                out.append((index, str(info.get("name", ""))))
        return out
    finally:
        audio.terminate()


def _enumerate_outputs() -> list[tuple[int, str]]:
    """Return [(index, name), ...] for all PyAudio devices with output channels.

    PyAudio is built on PortAudio and shares its device-index space with
    sounddevice, so an index resolved here is valid for sounddevice playback.
    """
    audio = pyaudio.PyAudio()
    try:
        out: list[tuple[int, str]] = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info.get("maxOutputChannels", 0) > 0:
                out.append((index, str(info.get("name", ""))))
        return out
    finally:
        audio.terminate()


def _default_output_index() -> int | None:
    audio = pyaudio.PyAudio()
    try:
        info = audio.get_default_output_device_info()
        return int(info.get("index")) if info else None
    except Exception:
        return None
    finally:
        audio.terminate()


def _find_respeaker(devices: list[tuple[int, str]]) -> tuple[int, str] | None:
    for index, name in devices:
        lo = name.lower()
        if "respeaker" in lo or "xvf3800" in lo:
            return index, name
    return None


def resolve_input_index(value: str | int) -> int:
    if isinstance(value, int):
        return value
    if value and str(value).strip().lower() != "auto":
        return int(value)

    # USB devices (the reSpeaker) sometimes enumerate a beat after pipeline
    # startup — and during a container *recreate* the outgoing container may
    # still hold the device for a second or two. Retry longer so we don't
    # give up on the real mic and land on a phantom node.
    last_devices: list[tuple[int, str]] = []
    for attempt in range(20):
        devices = _enumerate_inputs()
        last_devices = devices
        match = _find_respeaker(devices)
        if match is not None:
            index, name = match
            _log(f"[Audio] Auto-selected input device [{index}] {name}")
            return index
        if attempt < 19:
            time.sleep(1.0)

    # reSpeaker name match never hit. Don't blindly grab the first input —
    # on Jetson that is an HDMI/HDA/APE phantom node (0 real channels →
    # PortAudio -9998). Pick the most mic-like device instead, preferring USB
    # (which catches the reSpeaker even if its name flaked during a hotplug).
    _log("[Audio] reSpeaker / XVF3800 not found by name. Available inputs:")
    for index, name in last_devices:
        _log(f"[Audio]   [{index}] {name}")
    fb = _best_fallback_input(last_devices)
    if fb is not None:
        index, name = fb
        _log(
            f"[Audio] Auto-selected fallback input device [{index}] {name} "
            f"(skipped HDMI/HDA/APE phantom inputs)"
        )
        _log("[Audio] ⚠ Set MIC_INDEX env var explicitly if this is wrong.")
        return index

    # Absolute last resort: every input looked like a phantom node. Take the
    # first one so the pipeline can at least boot for diagnostics.
    if last_devices:
        index, name = last_devices[0]
        _log(f"[Audio] ⚠ no mic-like input found; using [{index}] {name}")
        return index

    raise RuntimeError("No audio input device found")


def resolve_output_index(value: str | int) -> int:
    """Resolve the TTS playback device index by reSpeaker name, mirroring
    resolve_input_index. Returns -1 to mean "let sounddevice use its default".

    PortAudio device indices are NOT stable across container restarts on
    Jetson (HDMI / APE virtual nodes shuffle ordering), so a hardcoded
    integer like SPEAKER_DEVICE=24 silently lands on a 0-output APE node and
    TTS plays into the void. Resolve by name the same way the mic is, so the
    speaker tracks the reSpeaker wherever it enumerates.
    """
    if isinstance(value, int):
        return value
    if value and str(value).strip().lower() != "auto":
        return int(value)

    last_devices: list[tuple[int, str]] = []
    for attempt in range(10):
        devices = _enumerate_outputs()
        last_devices = devices
        match = _find_respeaker(devices)
        if match is not None:
            index, name = match
            _log(f"[Audio] Auto-selected output device [{index}] {name}")
            return index
        if attempt < 9:
            time.sleep(1.0)

    _log("[Audio] reSpeaker / XVF3800 output not found after 10s. Available outputs:")
    for index, name in last_devices:
        _log(f"[Audio]   [{index}] {name}")
    default_index = _default_output_index()
    if default_index is not None:
        _log(f"[Audio] Falling back to system default output device [{default_index}]")
        _log("[Audio] ⚠ Set SPEAKER_DEVICE env var explicitly if TTS is silent.")
        return default_index
    _log("[Audio] No default output device; letting sounddevice choose.")
    return -1
