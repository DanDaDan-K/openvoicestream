"""Audio input/output device resolution — robust against PortAudio churn.

Background (why this was rewritten):
    The old implementation enumerated devices through PyAudio/PortAudio and
    matched the reSpeaker by name, falling back to "the first real-looking
    input" otherwise. On Jetson + Docker this proved unreliable for three
    independent reasons, all confirmed in the field:

      1. PortAudio caches its device list per process at the first
         ``Pa_Initialize`` and does NOT re-scan ALSA on a later re-init
         within the same process — so a retry loop sees the same stale list.
      2. A USB capture device that is *busy* (held by a dying container's
         TTS output) or *not yet enumerated* (cold-boot USB lag) is dropped
         from PortAudio's list entirely → the reSpeaker "disappears" and the
         old code fell back onto a Tegra APE phantom node (16 channels, opens
         fine, captures nothing → user hears "no response").
      3. PortAudio/ALSA card *indices* drift across reboots (the reSpeaker has
         been seen at index 0 and at index 24 on the same box), so a pinned
         integer ``MIC_INDEX`` silently points at the wrong card after a
         re-enumeration.

    The fix: resolve identity from ``/proc/asound/cards`` (ALSA-native — the
    card is listed there even while busy or before PortAudio can probe it),
    and hand sounddevice a *stable name substring* (e.g. ``"XVF3800"``, taken
    from the USB iProduct string, which never changes across reboots/ports).
    sounddevice resolves that substring to whatever index the device currently
    has at open time. No PortAudio enumeration, no index pinning, no phantom
    fallback.

Backward compatibility (``MIC_INDEX`` / ``SPEAKER_DEVICE`` accept):
    - integer or digit-string  -> passed through as a PortAudio index (e.g. ``24``)
    - any other non-"auto" str  -> passed through as a sounddevice name substring
                                   (e.g. ``reSpeaker`` — the documented form)
    - ``auto`` (default)        -> /proc-based reSpeaker detection -> name token
"""

import os
import re
import sys
import time

# Card descriptions in /proc/asound/cards that are NOT a real mic/speaker we
# ever want to auto-select: Jetson Tegra HDMI/HDA + APE virtual nodes, digital
# passthrough. Used only for the no-reSpeaker fallback so we never hand back a
# phantom node.
_PHANTOM_RE = re.compile(r"(?i)\bhdmi\b|\bhda\b|tegra|\bape\b|spdif|dummy")

# Stable identity tokens for the reSpeaker family (matched against the USB
# product string in /proc/asound/cards, case-insensitive). The first token that
# also appears verbatim in the PortAudio device name is preferred as the value
# handed to sounddevice.
_RESPEAKER_TOKENS = ("XVF3800", "reSpeaker", "4-Mic Array", "C16K6Ch")

# Line like: " 2 [Array          ]: USB-Audio - reSpeaker XVF3800 4-Mic Array"
_CARD_RE = re.compile(r"^\s*(\d+)\s+\[\s*([^\]]+?)\s*\]:\s*(.*)$")


def _log(msg: str) -> None:
    """Emit on stderr so the entrypoint (which captures stdout to read the
    resolved value) doesn't pick these informational lines up as the value."""
    print(msg, file=sys.stderr)


def _read_sound_cards() -> list[tuple[int, str, str]]:
    """Parse ``/proc/asound/cards`` -> ``[(card_no, card_id, description), ...]``.

    ALSA-native: a card is listed here even when its PCM is busy or PortAudio
    cannot open it, so this is immune to the enumeration gaps that break the
    PortAudio path.
    """
    try:
        with open("/proc/asound/cards", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:  # pragma: no cover - /proc always present on Linux
        _log(f"[Audio] cannot read /proc/asound/cards: {exc}")
        return []
    cards: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        m = _CARD_RE.match(line)
        if m:
            cards.append((int(m.group(1)), m.group(2), m.group(3).strip()))
    return cards


def _respeaker_card(cards: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    for no, cid, desc in cards:
        blob = f"{cid} {desc}".lower()
        if any(tok.lower() in blob for tok in _RESPEAKER_TOKENS):
            return (no, cid, desc)
    return None


def _name_token(card: tuple[int, str, str]) -> str:
    """Pick a stable substring sounddevice will match against the PortAudio
    device name (which embeds the USB product string)."""
    no, cid, desc = card
    blob = f"{cid} {desc}"
    for tok in _RESPEAKER_TOKENS:
        if tok.lower() in blob.lower():
            return tok
    # Fall back to the trailing product string ("... - reSpeaker XVF3800 ...").
    return desc.split(" - ", 1)[-1].strip() or cid


def _wait_for_respeaker(timeout_s: float) -> tuple[int, str, str] | None:
    """Poll /proc/asound/cards for the reSpeaker (handles cold-boot USB lag)."""
    attempts = max(1, int(timeout_s))
    for i in range(attempts):
        card = _respeaker_card(_read_sound_cards())
        if card is not None:
            return card
        if i < attempts - 1:
            time.sleep(1.0)
    return None


def _is_numeric(value: str) -> bool:
    return value.lstrip("-").isdigit()


def _first_real_card(cards: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    """A non-phantom card to fall back to when no reSpeaker is present
    (dev boxes / other mics). Never a Tegra/HDMI/APE node."""
    for card in cards:
        _, cid, desc = card
        if not _PHANTOM_RE.search(f"{cid} {desc}"):
            return card
    return None


def _resolve(value, *, kind: str) -> object:
    """Shared resolution for input ('mic') and output ('speaker')."""
    if isinstance(value, int):
        return value
    v = str(value).strip()
    if v.lower() != "auto":
        # Numeric -> PortAudio index (back-compat). Anything else -> name
        # substring passed straight to sounddevice (the documented form).
        return int(v) if _is_numeric(v) else v

    timeout = float(os.getenv("MIC_RESOLVE_TIMEOUT_S", "15"))
    card = _wait_for_respeaker(timeout)
    if card is not None:
        token = _name_token(card)
        _log(
            f"[Audio] {kind}: reSpeaker found in /proc/asound/cards "
            f"(card {card[0]} [{card[1]}]) -> name match '{token}'"
        )
        return token

    # No reSpeaker. Do NOT grab a phantom node. Prefer a real non-Tegra card by
    # name; otherwise hand back "" so sounddevice uses the system default.
    fallback = _first_real_card(_read_sound_cards())
    if fallback is not None:
        token = _name_token(fallback) if _respeaker_card([fallback]) else fallback[2] or fallback[1]
        _log(
            f"[Audio] ⚠ {kind}: reSpeaker not found after {timeout:.0f}s; "
            f"using real card '{token}'. Set MIC_INDEX/SPEAKER_DEVICE explicitly."
        )
        return token
    _log(
        f"[Audio] ⚠ {kind}: no reSpeaker and no real card found; "
        f"leaving to system default."
    )
    return ""


def resolve_input_index(value: "str | int") -> object:
    """Resolve ``MIC_INDEX`` to an int index OR a sounddevice name substring.

    Returns a value suitable for ``sounddevice``'s ``device=`` argument
    (and for ``sd.query_devices``), which accepts either an int index or a
    case-insensitive name substring. See module docstring for the rationale.
    """
    return _resolve(value, kind="mic")


def resolve_output_index(value: "str | int") -> object:
    """Resolve ``SPEAKER_DEVICE`` the same way as the mic, so TTS playback
    tracks the reSpeaker wherever it enumerates instead of landing on a
    0-output APE node and playing into the void."""
    return _resolve(value, kind="speaker")
