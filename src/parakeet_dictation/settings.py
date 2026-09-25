"""Persisted, menu-adjustable settings for the live-dictation streaming loop.

Settings are stored as plain JSON at SETTINGS_PATH and read once at startup
via load_settings(). The menu bar is the primary way to change them; the
legacy PARAKEET_* env vars still work too, but only as a one-off seed for
that particular launch (see load_settings) — they are never themselves
written back to the settings file.
"""

import json
import os
from pathlib import Path

# (context_size, depth) tuning for parakeet_mlx's transcribe_stream, plus how
# many seconds of audio to batch into each add_audio call.
#
# The two knobs that matter for accuracy are right_context (the second value
# in context_size — how many future encoder frames of local-attention lookahead
# each frame gets) and depth (how many encoder layers get cache that exactly
# matches a full non-streaming forward pass, vs. an approximation). Both cost
# about the same to run — a large fixed per-call overhead dominates regardless
# of chunk size or depth (measured) — so there's no real reason not to raise
# them, EXCEPT that parakeet_mlx computes how much audio must arrive before
# anything finalizes as drop_size = right_context * depth. A high enough
# product means nothing finalizes during any normal dictation, so the whole
# utterance stays in the volatile "draft" state the entire time. That's safe
# (never wipes what's on screen — see _apply_stream_result's shrink guard in
# main.py) but means more of the session stays eligible for a real revision,
# which is what more accuracy needs anyway. Bigger batches (fewer, larger
# add_audio calls) spend more of that fixed overhead on actual audio instead
# of wasting it, at the cost of text taking longer to first appear.
ACCURACY_PRESETS = {
    "fast": {"context_size": (256, 8), "depth": 2, "batch_seconds": 0.5},
    "balanced": {"context_size": (256, 16), "depth": 4, "batch_seconds": 0.5},
    "accurate": {"context_size": (256, 64), "depth": 12, "batch_seconds": 1.0},
}

# The push-to-talk trigger. "modifiers" is a list of specific pynput Key names
# (e.g. "alt_r", not generic "alt") and "key" is either None (a modifier-only
# combo — held keys start recording, releasing any of them stops it) or a
# single regular key's name/character (a chord — see main.py's _run_chord_listener
# for why chords can't preserve left/right specificity the way modifier-only
# combos can).
DEFAULT_HOTKEY = {"modifiers": ["alt_r"], "key": None}

# Spoken word -> literal symbol, applied to transcribed text before it's typed
# when "symbol_words_enabled" is on (see main.py's _apply_symbol_words). Only
# covers words that are unlikely to show up as themselves in ordinary prose —
# deliberately leaves out common nouns/punctuation-ish words like "period",
# "comma", "star", "dot", "bar", "hash" that would misfire constantly outside
# of deliberate symbol dictation.
SYMBOL_WORDS = {
    "tilde": "~",
    "backtick": "`",
    "underscore": "_",
    "ampersand": "&",
    "asterisk": "*",
    "caret": "^",
    "pipe": "|",
    "backslash": "\\",
    "forward slash": "/",
    "slash": "/",
    "hyphen": "-",
    "dash": "-",
    "equals sign": "=",
    "equal sign": "=",
    "plus sign": "+",
    "at sign": "@",
    "at symbol": "@",
    "pound sign": "#",
    "hash sign": "#",
    "hashtag": "#",
    "percent sign": "%",
    "dollar sign": "$",
    "open paren": "(",
    "open parenthesis": "(",
    "close paren": ")",
    "close parenthesis": ")",
    "open bracket": "[",
    "close bracket": "]",
    "open brace": "{",
    "open curly": "{",
    "close brace": "}",
    "close curly": "}",
    "less than": "<",
    "greater than": ">",
    "double quote": '"',
    "single quote": "'",
}

DEFAULTS = {
    "accuracy_mode": "accurate",
    "llm_enabled": False,
    "symbol_words_enabled": False,
    "verbose_logging": False,
    "hotkey": DEFAULT_HOTKEY,
}


def _is_valid_hotkey(hotkey) -> bool:
    return (
        isinstance(hotkey, dict)
        and isinstance(hotkey.get("modifiers"), list)
        and len(hotkey["modifiers"]) > 0
        and all(isinstance(m, str) for m in hotkey["modifiers"])
        and (hotkey.get("key") is None or isinstance(hotkey["key"], str))
    )

SETTINGS_PATH = (
    Path.home() / "Library" / "Application Support" / "parakeet-dictation" / "settings.json"
)


def load_settings() -> dict:
    """Load persisted settings, seeded/overridden by explicit env vars for this launch."""
    settings = dict(DEFAULTS)
    settings["hotkey"] = {"modifiers": list(DEFAULT_HOTKEY["modifiers"]), "key": DEFAULT_HOTKEY["key"]}
    try:
        if SETTINGS_PATH.exists():
            saved = json.loads(SETTINGS_PATH.read_text())
            settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except Exception:
        pass  # corrupt/unreadable settings file — fall back to defaults rather than crash

    if os.environ.get("PARAKEET_ENABLE_LLM") == "1":
        settings["llm_enabled"] = True
    if os.environ.get("PARAKEET_LOG", "").lower() in ("info", "debug"):
        settings["verbose_logging"] = True

    if settings["accuracy_mode"] not in ACCURACY_PRESETS:
        settings["accuracy_mode"] = DEFAULTS["accuracy_mode"]
    if not _is_valid_hotkey(settings["hotkey"]):
        settings["hotkey"] = {"modifiers": list(DEFAULT_HOTKEY["modifiers"]), "key": DEFAULT_HOTKEY["key"]}

    return settings


def save_settings(settings: dict) -> None:
    """Best-effort persist. A failed write just means this change won't survive a relaunch."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        to_save = {k: settings[k] for k in DEFAULTS}
        SETTINGS_PATH.write_text(json.dumps(to_save, indent=2))
    except Exception:
        pass
