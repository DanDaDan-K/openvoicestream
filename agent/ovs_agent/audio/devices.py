import sys
import time

import pyaudio


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


def _default_input_index() -> int | None:
    audio = pyaudio.PyAudio()
    try:
        info = audio.get_default_input_device_info()
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


def _find_name(devices: list[tuple[int, str]], needle: str) -> tuple[int, str] | None:
    if needle.lower() == "respeaker":
        return _find_respeaker(devices)
    needle = needle.lower()
    for index, name in devices:
        if needle in name.lower():
            return index, name
    return None


def _requested_name(value: str | int | None) -> str | None:
    if isinstance(value, int):
        return None
    raw = "" if value is None else str(value).strip()
    if not raw or raw.lower() == "auto":
        return "reSpeaker"
    if raw.lower() in ("default", "system", "system_default"):
        return None
    try:
        int(raw)
    except ValueError:
        return raw
    return None


def _explicit_index(value: str | int | None) -> int | None:
    if isinstance(value, int):
        return value
    raw = "" if value is None else str(value).strip()
    if not raw or raw.lower() in ("auto", "default", "system", "system_default"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_input_index(value: str | int | None) -> int:
    explicit = _explicit_index(value)
    if explicit is not None:
        return explicit
    raw = "" if value is None else str(value).strip().lower()
    requested = None if raw in ("default", "system", "system_default") else (
        _requested_name(value) or "reSpeaker"
    )

    # USB devices (the reSpeaker) sometimes enumerate a beat after pipeline
    # startup. Try a few times before falling back to whatever is available
    # so we don't get stuck on the Jetson APE every cold boot.
    last_devices: list[tuple[int, str]] = []
    if requested is not None:
        for attempt in range(10):
            devices = _enumerate_inputs()
            last_devices = devices
            match = _find_name(devices, requested)
            if match is not None:
                index, name = match
                _log(f"[Audio] Auto-selected input device [{index}] {name}")
                return index
            if attempt < 9:
                time.sleep(1.0)

        _log(f"[Audio] input device matching {requested!r} not found after 10s. Available inputs:")
        for index, name in last_devices:
            _log(f"[Audio]   [{index}] {name}")
    default_index = _default_input_index()
    if default_index is not None:
        _log(f"[Audio] Falling back to system default input device [{default_index}]")
        _log("[Audio] ⚠ Set MIC_INDEX env var explicitly if ASR is silent.")
        return default_index

    _log("[Audio] No default input device; letting sounddevice choose.")
    return -1


def resolve_output_index(value: str | int | None) -> int:
    """Resolve the TTS playback device index by reSpeaker name, mirroring
    resolve_input_index. Returns -1 to mean "let sounddevice use its default".

    PortAudio device indices are NOT stable across container restarts on
    Jetson (HDMI / APE virtual nodes shuffle ordering), so a hardcoded
    integer like SPEAKER_DEVICE=24 silently lands on a 0-output APE node and
    TTS plays into the void. Resolve by name the same way the mic is, so the
    speaker tracks the reSpeaker wherever it enumerates.
    """
    explicit = _explicit_index(value)
    if explicit is not None:
        return explicit
    raw = "" if value is None else str(value).strip().lower()
    requested = None if raw in ("default", "system", "system_default") else (
        _requested_name(value) or "reSpeaker"
    )

    last_devices: list[tuple[int, str]] = []
    if requested is not None:
        for attempt in range(10):
            devices = _enumerate_outputs()
            last_devices = devices
            match = _find_name(devices, requested)
            if match is not None:
                index, name = match
                _log(f"[Audio] Auto-selected output device [{index}] {name}")
                return index
            if attempt < 9:
                time.sleep(1.0)

        _log(f"[Audio] output device matching {requested!r} not found after 10s. Available outputs:")
        for index, name in last_devices:
            _log(f"[Audio]   [{index}] {name}")
    default_index = _default_output_index()
    if default_index is not None:
        _log(f"[Audio] Falling back to system default output device [{default_index}]")
        _log("[Audio] ⚠ Set SPEAKER_DEVICE env var explicitly if TTS is silent.")
        return default_index
    _log("[Audio] No default output device; letting sounddevice choose.")
    return -1
