#!/usr/bin/env python3
import os
import io
import time
import tempfile
import threading
import concurrent.futures
import queue
import pyaudio
import wave
import numpy as np
import rumps
from pynput import keyboard
from pynput.keyboard import Controller
from parakeet_mlx import from_pretrained
import signal
from .text_selection import TextSelection
from .logger_config import setup_logging
import re
from .settings import ACCURACY_PRESETS, DEFAULT_HOTKEY, SYMBOL_WORDS, load_settings, save_settings
from mlx_lm import load as mlx_load, generate as mlx_generate
import argparse

# Set HuggingFace tokenizers parallelism warning off
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Logging: default to WARNING (lower overhead), override via env ----------
logger = setup_logging()
_log_env = os.getenv("PARAKEET_LOG", "").lower()
if _log_env in ("debug", "info", "warning", "error", "critical"):
    import logging as _logging
    logger.setLevel(getattr(_logging, _log_env.upper()))
else:
    import logging as _logging
    logger.setLevel(_logging.WARNING)

# Set up a global flag for handling SIGINT
exit_flag = False

def signal_handler(signum, frame):  # FIXED: accept (signum, frame)
    """Global signal handler for graceful shutdown"""
    global exit_flag
    logger.info("Shutdown signal received, exiting gracefully...")
    exit_flag = True
    threading.Timer(2.0, lambda: os._exit(0)).start()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Sentinel put onto a recording's own stream queue to mark "no more audio is
# coming for this session" — see start_recording / stop_recording / _stream_feed_loop.
_STREAM_STOP = object()

# Auto-stop a held-open recording if nothing at or above this RMS arrives for
# this long — guards against a missed key-up event (or a mic that's gone
# silent) leaving the recording, and the hotkey, stuck open indefinitely.
SILENCE_TIMEOUT_SECS = float(os.getenv("PARAKEET_SILENCE_TIMEOUT_SECS", "15"))
_SILENCE_RMS_THRESHOLD = 5e-4  # same floor _stream_feed_loop uses for auto-gain

LLM_MENU_TITLE = "Enable Text Editing (Qwen)"

# Longest phrase first so e.g. "open parenthesis" matches whole rather than
# leaving "open " to fall through — though \b boundaries below already stop
# "open paren" from matching inside "open parenthesis" on their own.
_SYMBOL_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(SYMBOL_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _apply_symbol_words(text: str) -> str:
    """Replace spoken symbol names ("tilde", "open paren", ...) with the
    literal symbol, per SYMBOL_WORDS. Gated behind settings["symbol_words_enabled"]
    since it's a real tradeoff against ordinary prose dictation, not a pure win."""
    return _SYMBOL_WORD_PATTERN.sub(lambda m: SYMBOL_WORDS[m.group(0).lower()], text)


def _is_model_cached(repo_id: str) -> bool:
    """Whether repo_id is already fully present in the local Hugging Face cache,
    checked with no network access. Used to warn about the ~1GB Qwen download
    in the menu item's title before the user opts into it."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False


# ---------------------------
# Hotkey key <-> name helpers
#
# A captured/stored hotkey is a plain dict: {"modifiers": [<pynput Key name>, ...],
# "key": <character or special-key name, or None>}.
#
# IMPORTANT, verified against pynput's actual macOS Key enum (not assumed): only
# the RIGHT-side modifier is a distinct enum member (alt_r, ctrl_r, cmd_r,
# shift_r). "alt_l" etc. are not real — pynput's Key.alt_l is literally an alias
# for the generic Key.alt, which is what a LEFT modifier press actually reports
# as. So the true set of distinguishable modifier identities on this platform is
# {alt, alt_r, ctrl, ctrl_r, cmd, cmd_r, shift, shift_r} — the bare name means
# "left, or unspecified/either," never a separate "_l" name.
# ---------------------------

_MODIFIER_NAMES = {
    "alt", "alt_r", "ctrl", "ctrl_r", "cmd", "cmd_r", "shift", "shift_r",
}
_MODIFIER_FAMILY = {  # for building a GlobalHotKeys chord token: <alt>, <ctrl>, etc.
    "alt": "alt", "alt_r": "alt",
    "ctrl": "ctrl", "ctrl_r": "ctrl",
    "cmd": "cmd", "cmd_r": "cmd",
    "shift": "shift", "shift_r": "shift",
}
_DISPLAY_NAMES = {
    "alt": "Option", "alt_r": "Right Option",
    "ctrl": "Control", "ctrl_r": "Right Control",
    "cmd": "Command", "cmd_r": "Right Command",
    "shift": "Shift", "shift_r": "Right Shift",
}


def _is_modifier_key(key) -> bool:
    return isinstance(key, keyboard.Key) and key.name in _MODIFIER_NAMES


def _key_to_name(key):
    """A pynput Key/KeyCode -> the string this app stores it as, or None if
    it's some other special key (e.g. Escape) not meant to be bindable."""
    if isinstance(key, keyboard.Key):
        return key.name
    if isinstance(key, keyboard.KeyCode) and key.char:
        return key.char.lower()
    return None


def _name_to_key(name):
    """The reverse of _key_to_name."""
    special = getattr(keyboard.Key, name, None)
    if isinstance(special, keyboard.Key):
        return special
    return keyboard.KeyCode.from_char(name)


def _hotkey_display_name(hotkey: dict) -> str:
    parts = [_DISPLAY_NAMES.get(m, m) for m in hotkey["modifiers"]]
    if hotkey.get("key"):
        parts.append(hotkey["key"].upper())
    return " + ".join(parts)

class WhisperDictationApp(rumps.App):
    def __init__(self):
        super(WhisperDictationApp, self).__init__("🎙️", quit_button=rumps.MenuItem("Quit"))
        self.settings = load_settings()

        self.status_item = rumps.MenuItem("Status: Ready")
        self.recording_menu_item = rumps.MenuItem("Start Recording")

        accuracy_menu = rumps.MenuItem("Accuracy")
        self.accuracy_items = {}
        for mode in ACCURACY_PRESETS:
            item = rumps.MenuItem(mode.capitalize(), callback=self.set_accuracy_mode)
            item.state = mode == self.settings["accuracy_mode"]
            self.accuracy_items[mode] = item
            accuracy_menu.add(item)

        self.llm_config = {
            "model_id": os.getenv("PARAKEET_LLM_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
            "max_tokens": int(os.getenv("PARAKEET_LLM_MAX_TOKENS", "192")),
            "temperature": float(os.getenv("PARAKEET_LLM_TEMP", "0.2")),
            "top_p": float(os.getenv("PARAKEET_LLM_TOP_P", "0.9")),
        }
        llm_title = LLM_MENU_TITLE
        if not _is_model_cached(self.llm_config["model_id"]):
            llm_title += " (downloads ~1GB)"
        self.llm_item = rumps.MenuItem(llm_title, callback=self.toggle_llm)
        self.llm_item.state = self.settings["llm_enabled"]

        self.verbose_item = rumps.MenuItem("Verbose Logging", callback=self.toggle_verbose_logging)
        self.verbose_item.state = self.settings["verbose_logging"]

        self.symbol_words_item = rumps.MenuItem('Type Symbols ("tilde" → ~)', callback=self.toggle_symbol_words)
        self.symbol_words_item.state = self.settings["symbol_words_enabled"]

        self.hotkey_item = rumps.MenuItem(
            f"Hotkey: {_hotkey_display_name(self.settings['hotkey'])}",
        )
        self.customize_hotkey_item = rumps.MenuItem("Customize...", callback=self.customize_hotkey)
        self.reset_hotkey_item = rumps.MenuItem("Reset to Default (Right Option)", callback=self.reset_hotkey)
        self.hotkey_item.add(self.customize_hotkey_item)
        self.hotkey_item.add(self.reset_hotkey_item)

        self.menu = [
            self.recording_menu_item,
            None,
            accuracy_menu,
            self.llm_item,
            self.symbol_words_item,
            self.verbose_item,
            self.hotkey_item,
            None,
            self.status_item,
        ]
        if self.settings["verbose_logging"] and not _log_env:
            logger.setLevel(_logging.INFO)

        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.keyboard_controller = Controller()
        self.text_selector = TextSelection()

        # Live-dictation state (see start_recording / _stream_feed_loop).
        # False whenever a selection turns this into a Qwen edit instruction instead.
        self._live_mode = False
        self._typed_finalized = ""
        self._typed_draft = ""

        # Initialize Parakeet model (async).
        # All MLX calls (load + every transcription) must run on this single
        # persistent worker thread: MLX ties lazily-evaluated arrays to the
        # stream of the thread that created them, so loading the model on one
        # thread and transcribing on a fresh thread each time raises
        # "There is no Stream(cpu, 1) in current thread."
        self.model = None
        self.mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-worker"
        )
        self.mlx_executor.submit(self.load_model)

        # NEW: Initialize Qwen (MLX) editor model (async)
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_ready = False
        self.llm_loading = False
        # The speak-an-edit rewrite loads Qwen2.5-1.5B and holds it resident for a
        # feature that isn't always wanted; off by default (menu bar toggle, or
        # PARAKEET_ENABLE_LLM=1 to start with it already on). Routed through
        # mlx_executor like every other MLX call — mlx_lm is MLX too, so loading it
        # on an ad-hoc thread and generating from it later on mlx_executor's worker
        # would hit the same cross-thread Stream crash the ASR model has (see the
        # comment above self.mlx_executor).
        if self.settings["llm_enabled"]:
            self.mlx_executor.submit(self.load_llm)
        else:
            logger.info("LLM rewrite disabled (menu bar, or PARAKEET_ENABLE_LLM=1 to start enabled)")

        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 512  # smaller chunk -> snappier stop

        # Hotkey state
        self.is_recording_with_hotkey = False
        self._active_hotkey_stoppers = []
        self._capturing_hotkey = False

        # Set up the configurable global push-to-talk hotkey
        self.setup_global_monitor()

        logger.info("Started WhisperDictation app. Look for 🎙️ in your menu bar.")
        logger.info(f"Press and HOLD {_hotkey_display_name(self.settings['hotkey'])} to record. Release to transcribe.")
        logger.info("Press Ctrl+C to quit the application.")
        logger.info("If hotkeys don’t fire: System Settings → Privacy & Security → Accessibility + Input Monitoring")

        self.watchdog = threading.Thread(target=self.check_exit_flag, daemon=True)
        self.watchdog.start()

    def check_exit_flag(self):
        while True:
            if exit_flag:
                logger.info("Watchdog detected exit flag, shutting down...")
                self.cleanup()
                rumps.quit_application()
                os._exit(0)
            time.sleep(0.5)

    def cleanup(self):
        logger.info("Cleaning up resources...")
        self.recording = False
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            try:
                self.recording_thread.join(timeout=1.0)
            except Exception:
                pass
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except Exception:
                pass

    def load_model(self):
        self.title = "🎙️ (Loading...)"
        self.status_item.title = "Status: Loading Parakeet model..."
        try:
            model_id = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
            self.model = from_pretrained(model_id)

            # Warm-up: run a tiny silent clip once to trigger JIT/graph compilation & caches
            try:
                sr = 16000
                silence = (np.zeros(int(0.3 * sr)).astype(np.int16)).tobytes()
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # int16
                    wf.setframerate(sr)
                    wf.writeframes(silence)
                buf.seek(0)
                _ = self.model.transcribe(buf)
                logger.info("Parakeet warm-up done")
            except Exception as we:
                logger.debug(f"Warm-up skipped/fallback due to: {we}")

            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
            logger.info("Parakeet model loaded successfully!")
        except Exception as e:
            self.title = "🎙️ (Error)"
            self.status_item.title = "Status: Error loading model"
            logger.error(f"Error loading Parakeet model: {e}")

    # NEW: Load Qwen (MLX) LLM editor
    def load_llm(self):
        if mlx_load is None or mlx_generate is None:
            logger.warning("mlx-lm not installed; local LLM edits disabled. `pip install mlx-lm` to enable.")
            return
        self.llm_loading = True
        try:
            logger.info(f"Loading Qwen MLX model: {self.llm_config['model_id']}")
            # Many Qwen MLX models need trust_remote_code; eos token is usually set in tokenizer config.
            self.llm_model, self.llm_tokenizer = mlx_load(
                self.llm_config["model_id"],
                tokenizer_config={"trust_remote_code": True},
            )
            # Warm-up to prime kernels/caches
            try:
                _ = self.enhance_with_qwen("return the same text", "warmup")
                logger.info("Qwen MLX warm-up done")
            except Exception as we:
                logger.debug(f"Qwen warm-up skipped due to: {we}")
            self.llm_ready = True
            logger.info("Qwen MLX model ready for local edits")
            self.llm_item.title = LLM_MENU_TITLE  # drop the "(downloads ~1GB)" note — it's cached now
        except Exception as e:
            logger.error(f"Failed to load Qwen MLX model: {e}")
            self.llm_ready = False
        finally:
            self.llm_loading = False

    # ---------------------------
    # Global hotkey + release monitor
    #
    # Configurable via the Hotkey menu (Customize.../Reset to Default), stored as
    # self.settings["hotkey"]. Two different pynput mechanisms depending on what's
    # configured, chosen automatically:
    #   - Modifier-only combo (no regular key): _run_modifier_only_listener, a
    #     generalization of the original hardcoded right-Option-only behavior —
    #     start when every required modifier is held, stop when any releases.
    #     Preserves left/right specificity.
    #   - Modifier(s) + a regular key (a "chord", e.g. Ctrl+Option+A):
    #     _run_chord_listener, via GlobalHotKeys — the same mechanism upstream
    #     used for its original Ctrl+Alt+A hotkey. GlobalHotKeys canonicalizes
    #     modifiers to their generic (side-independent) form before matching
    #     (verified against pynput's own HotKey.canonical/parse), so a captured
    #     "right Option" collapses to "either Option" for chords specifically —
    #     unavoidable with this library, not a bug here.
    # Both run inside _hotkey_loop, which rebuilds from current settings every
    # time a listener stops — including a deliberate stop from customize_hotkey/
    # reset_hotkey, which is how a rebind takes effect without an app restart.
    # ---------------------------
    def setup_global_monitor(self):
        self.key_monitor_thread = threading.Thread(target=self._hotkey_loop, daemon=True)
        self.key_monitor_thread.start()

    def _hotkey_loop(self):
        while not exit_flag:
            try:
                self._run_hotkey_listener()
            except Exception as e:
                logger.error(f"Error with keyboard listeners: {e}")
                logger.error("Please check Accessibility/Input Monitoring permissions in System Settings.")
                return
            # _capture_hotkey_thread stops us and runs its own temporary listener
            # while it works; wait for it to finish (and write new settings)
            # rather than immediately rebuilding the old hotkey out from under it.
            while self._capturing_hotkey and not exit_flag:
                time.sleep(0.1)

    def _run_hotkey_listener(self):
        hotkey = self.settings["hotkey"]
        logger.info(f"Global hotkey listener: hold {_hotkey_display_name(hotkey)} to record")
        if hotkey.get("key"):
            self._run_chord_listener(hotkey["modifiers"], hotkey["key"])
        else:
            self._run_modifier_only_listener(hotkey["modifiers"])

    def _run_modifier_only_listener(self, modifier_names):
        # on_press/on_release below MUST return almost instantly: pynput's macOS
        # backend runs them synchronously inside a CGEventTap callback, and the
        # WindowServer blocks system-wide keyboard delivery until that callback
        # returns. start_recording/stop_recording can block for a while (thread
        # joins, and — when text-editing mode is on — a synthetic Cmd+C plus
        # ~0.3s of sleeps to read the selection) and that synthetic keystroke is
        # itself a re-entrant CGEventPost from inside a live tap callback, which
        # is its own deadlock hazard. Once observed to freeze the keyboard
        # system-wide, requiring a hard shutdown. Always hand off to a thread
        # here instead of calling them directly.
        required = {_name_to_key(name) for name in modifier_names}
        held = set()

        def on_press(key):
            if key in required:
                held.add(key)
                if held == required and not self.recording and not self.is_recording_with_hotkey:
                    self.is_recording_with_hotkey = True
                    logger.info(f"STARTING recording via {' + '.join(modifier_names)}")
                    threading.Thread(target=self.start_recording, daemon=True).start()

        def on_release(key):
            if key in required:
                held.discard(key)
                if self.is_recording_with_hotkey and self.recording:
                    logger.info(f"STOPPING recording via {' + '.join(modifier_names)} release")
                    self.is_recording_with_hotkey = False
                    threading.Thread(target=self.stop_recording, daemon=True).start()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._active_hotkey_stoppers = [listener.stop]
        with listener:
            listener.join()

    def _run_chord_listener(self, modifier_names, key_name):
        families = sorted({_MODIFIER_FAMILY[m] for m in modifier_names})
        chord_str = "+".join(f"<{f}>" for f in families) + "+" + key_name

        def start():
            if not self.recording and not self.is_recording_with_hotkey:
                self.is_recording_with_hotkey = True
                logger.info(f"STARTING recording via {chord_str}")
                threading.Thread(target=self.start_recording, daemon=True).start()

        # Release-detection watches both the generic (left/either) and the
        # right-specific form of every modifier family involved, since a real
        # keypress reports as one of those even though the chord itself only
        # matched on the generic family.
        release_keys = set()
        for family in families:
            release_keys.add(getattr(keyboard.Key, family, None))
            release_keys.add(getattr(keyboard.Key, f"{family}_r", None))
        release_keys.discard(None)

        def on_release(key):
            if key in release_keys and self.is_recording_with_hotkey and self.recording:
                logger.info(f"STOPPING recording via {chord_str} release")
                self.is_recording_with_hotkey = False
                threading.Thread(target=self.stop_recording, daemon=True).start()

        hotkeys = keyboard.GlobalHotKeys({chord_str: start})
        release_listener = keyboard.Listener(on_release=on_release)
        self._active_hotkey_stoppers = [hotkeys.stop, release_listener.stop]
        with hotkeys, release_listener:
            hotkeys.join()

    def customize_hotkey(self, sender):
        if self._capturing_hotkey:
            return
        self._capturing_hotkey = True
        threading.Thread(target=self._capture_hotkey_thread, daemon=True).start()

    def _capture_hotkey_thread(self):
        for stop in self._active_hotkey_stoppers:
            try:
                stop()
            except Exception:
                pass

        self.title = "🎙️ (Press new hotkey...)"
        self.status_item.title = "Status: Press and hold your new hotkey, then release..."
        logger.info("Capture mode: press and hold your desired hotkey, then release it.")

        try:
            while True:
                held = set()
                captured = set()

                def on_press(key):
                    held.add(key)
                    captured.update(held)

                def on_release(key):
                    return False  # stop the listener on the first release

                with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                    listener.join()

                if not captured:
                    continue

                modifiers = [k for k in captured if _is_modifier_key(k)]
                regular = [k for k in captured if not _is_modifier_key(k)]

                if len(regular) > 1:
                    logger.warning("Captured more than one non-modifier key — hold just one regular key (plus modifiers), try again.")
                    self.status_item.title = "Status: Only one regular key allowed — try again"
                    continue
                if regular and not modifiers:
                    logger.warning("A bare key with no modifier would type into whatever you're focused on — hold at least one modifier too, try again.")
                    self.status_item.title = "Status: Need at least one modifier — try again"
                    continue
                if not modifiers:
                    continue

                mod_names = [n for n in (_key_to_name(k) for k in modifiers) if n]
                key_name = _key_to_name(regular[0]) if regular else None
                if regular and key_name is None:
                    logger.warning("Didn't recognize that key — try again.")
                    continue

                self.settings["hotkey"] = {"modifiers": mod_names, "key": key_name}
                save_settings(self.settings)
                break
        finally:
            self._capturing_hotkey = False
            self.title = "🎙️"

        display = _hotkey_display_name(self.settings["hotkey"])
        self.hotkey_item.title = f"Hotkey: {display}"
        self.status_item.title = "Status: Ready"
        logger.info(f"Hotkey bound to {display}")
        try:
            rumps.notification("Parakeet Dictation", "Hotkey updated", f"Bound to {display}")
        except Exception:
            pass

    def reset_hotkey(self, sender):
        self.settings["hotkey"] = {"modifiers": list(DEFAULT_HOTKEY["modifiers"]), "key": DEFAULT_HOTKEY["key"]}
        save_settings(self.settings)
        self.hotkey_item.title = f"Hotkey: {_hotkey_display_name(self.settings['hotkey'])}"
        logger.info("Hotkey reset to default (right Option)")
        for stop in self._active_hotkey_stoppers:
            try:
                stop()
            except Exception:
                pass

    # ---------------------------
    # Menu item click
    # ---------------------------
    @rumps.clicked("Start Recording")
    def toggle_recording(self, sender):
        if not self.recording:
            self.start_recording()
            sender.title = "Stop Recording"
        else:
            self.stop_recording()
            sender.title = "Start Recording"

    def set_accuracy_mode(self, sender):
        mode = next(m for m, item in self.accuracy_items.items() if item is sender)
        for item in self.accuracy_items.values():
            item.state = False
        sender.state = True
        self.settings["accuracy_mode"] = mode
        save_settings(self.settings)
        logger.info(f"Accuracy mode set to {mode} (takes effect on your next recording)")

    def toggle_llm(self, sender):
        enabled = not sender.state
        sender.state = enabled
        self.settings["llm_enabled"] = enabled
        save_settings(self.settings)
        if enabled:
            if self.llm_ready:
                logger.info("Text editing enabled")
            elif self.llm_loading:
                logger.info("Text-edit model is already loading")
            else:
                logger.info("Loading Qwen for text editing (enabled from menu)...")
                self.status_item.title = "Status: Loading text-edit model..."
                self.mlx_executor.submit(self.load_llm)
        else:
            logger.info("Text editing disabled")

    def toggle_verbose_logging(self, sender):
        enabled = not sender.state
        sender.state = enabled
        self.settings["verbose_logging"] = enabled
        save_settings(self.settings)
        logger.setLevel(_logging.INFO if enabled else _logging.WARNING)

    def toggle_symbol_words(self, sender):
        enabled = not sender.state
        sender.state = enabled
        self.settings["symbol_words_enabled"] = enabled
        save_settings(self.settings)
        logger.info(f"Symbol words {'enabled' if enabled else 'disabled'}")

    # ---------------------------
    # Recording & transcription
    # ---------------------------
    def start_recording(self):
        if not hasattr(self, 'model') or self.model is None:
            logger.warning("Model not loaded. Please wait for the model to finish loading.")
            self.status_item.title = "Status: Waiting for model to load"
            return

        self.frames = []
        self.recording = True
        self._last_sound_time = time.time()
        self.title = "🎙️ (Recording)"
        self.status_item.title = "Status: Recording..."
        logger.info("Recording started. Speak now...")

        # Text selected before you start talking means "speak an edit instruction for
        # this selection" (Qwen path below) — that stays batch-transcribed on release.
        # Otherwise, type live as you talk via the MLX streaming decoder. If text
        # editing is disabled, a selection no longer means anything special, so stay
        # live rather than needlessly taking the slower batch path — and, just as
        # importantly, skip calling get_selected_text() at all in that case: it
        # simulates Cmd+C (press synthetic Cmd, tap 'c', release synthetic Cmd), and
        # if your hotkey itself involves physically holding Cmd, that synthetic
        # Cmd-release looks identical to you actually letting go — instantly firing
        # a false stop before any real audio is captured.
        has_selection = self.settings["llm_enabled"] and bool(self.text_selector.get_selected_text())
        self._live_mode = not (has_selection and self.settings["llm_enabled"])
        if self._live_mode:
            self._stream_queue = queue.Queue()

        # Use a callback stream for near-instant stop
        self.recording_thread = threading.Thread(target=self._record_audio_callback_loop, daemon=True)
        self.recording_thread.start()
        threading.Thread(target=self._silence_watchdog, daemon=True).start()

        if self._live_mode:
            # Pass this session's queue explicitly rather than letting the loop
            # read self._stream_queue on every iteration: if you start a new
            # recording again before this session's loop has finished draining
            # (mlx_executor has one worker, so a slow drain queues the next
            # session behind it), self._stream_queue gets reassigned to the new
            # session's queue out from under the still-running old loop.
            self.mlx_executor.submit(self._stream_feed_loop, self._stream_queue)

    def _record_audio_callback_loop(self):
        def _cb(in_data, frame_count, time_info, status_flags):
            # in_data is bytes for paInt16 mono frames
            if self.recording:
                self.frames.append(in_data)
                if self._live_mode:
                    self._stream_queue.put(in_data)
                # Cheap RMS check to feed the silence watchdog. Runs on
                # PortAudio's own realtime callback thread, so this has to
                # stay fast and non-blocking — no logging, no locks.
                pcm16 = np.frombuffer(in_data, dtype=np.int16)
                if pcm16.size:
                    rms = float(np.sqrt(np.mean(np.square(pcm16.astype(np.float32) / 32768.0))))
                    if rms > _SILENCE_RMS_THRESHOLD:
                        self._last_sound_time = time.time()
                return (None, pyaudio.paContinue)
            else:
                return (None, pyaudio.paComplete)

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=_cb
        )
        try:
            stream.start_stream()
            while stream.is_active():
                if not self.recording:
                    break
                time.sleep(0.01)
        finally:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _silence_watchdog(self):
        """Auto-stops a recording session that's gone SILENCE_TIMEOUT_SECS
        without hearing anything above _SILENCE_RMS_THRESHOLD — e.g. a
        missed key-up event, or the mic silently failing, leaving the
        hotkey stuck "held" forever. Runs on its own thread (never the
        hotkey listener's tap-callback thread or recording_thread itself,
        which stop_recording's join() would deadlock against), so it's
        safe for it to call stop_recording() directly.
        """
        session = self.recording_thread
        while self.recording and self.recording_thread is session:
            if time.time() - self._last_sound_time > SILENCE_TIMEOUT_SECS:
                logger.warning(f"No audio detected for {SILENCE_TIMEOUT_SECS:.0f}s — auto-stopping recording")
                self.stop_recording()
                return
            time.sleep(0.5)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=5)
            if self.recording_thread.is_alive():
                logger.error("recording_thread didn't stop within 5s (stuck closing the audio stream?) — continuing without waiting further")

        if self._live_mode:
            # self.recording_thread.join() above guarantees every real audio chunk
            # for this session is already queued, so it's safe to enqueue the stop
            # marker now — the queue's FIFO order keeps it after all of them.
            # _stream_feed_loop (already running on the mlx worker, possibly still
            # behind a previous session's queued work) drains up through this
            # marker, does a final reconciliation pass, and sets the status itself.
            self._stream_queue.put(_STREAM_STOP)
            self.status_item.title = "Status: Finishing..."
            return

        self.title = "🎙️ (Transcribing)"
        self.status_item.title = "Status: Transcribing..."
        logger.info("Recording stopped. Transcribing...")

        self.mlx_executor.submit(self.process_recording)

    def _apply_stream_result(self, stream):
        """Type the delta between what's on screen and the streaming decoder's
        current state.

        stream.finalized_tokens is only ever grown via .extend() (parakeet_mlx
        never rewrites or drops earlier entries), so text built from it is
        provably append-only — safe to type once and never touch again. Only
        stream.draft_tokens (the small unconfirmed tail, replaced wholesale
        each update) is backspaced and retyped. An earlier version diffed the
        combined text directly, which broke when the decoder revised
        something near the START of the utterance (e.g. capitalizing the
        first word once more context arrived): the common-prefix comparison
        found no match from character 0 and backspaced the entire sentence.

        Deliberately not stripping the leading space each new recording
        session naturally starts with (SentencePiece tokens carry it): that
        space is what separates this dictation burst from whatever you
        dictated right before it. The only cost is a single leading space if
        this is the very first thing typed into an empty field.

        Also never lets the total displayed length shrink. draft_tokens can
        legitimately get shorter between updates — most visibly right at
        release, when the last sliver of trailing audio (silence, a breath)
        enters the decoder's local-attention window and it reconsiders
        whatever was still in draft, sometimes down to nothing. Once you've
        stopped talking, no further audio is ever coming to justify that
        revision, so the last good hypothesis is kept on screen instead.
        """
        finalized_text = "".join(t.text for t in stream.finalized_tokens)
        draft_text = "".join(t.text for t in stream.draft_tokens)
        if self.settings["symbol_words_enabled"]:
            # Applied here (before the length comparisons below) so every
            # length/prefix check downstream operates on the same
            # already-substituted strings consistently across updates.
            finalized_text = _apply_symbol_words(finalized_text)
            draft_text = _apply_symbol_words(draft_text)

        if len(finalized_text) + len(draft_text) < len(self._typed_finalized) + len(self._typed_draft):
            return

        if self._typed_draft:
            for _ in range(len(self._typed_draft)):
                self.keyboard_controller.press(keyboard.Key.backspace)
                self.keyboard_controller.release(keyboard.Key.backspace)

        if len(finalized_text) > len(self._typed_finalized):
            self.keyboard_controller.type(finalized_text[len(self._typed_finalized):])
            self._typed_finalized = finalized_text

        if draft_text:
            self.keyboard_controller.type(draft_text)
        self._typed_draft = draft_text

    def _stream_feed_loop(self, stream_queue):
        """Runs on the persistent mlx_executor worker for the whole recording:
        pulls raw mic chunks off stream_queue, batches ~0.5s at a time into the
        MLX streaming decoder, and types the delta after each update.

        stream_queue is passed explicitly (not read as self._stream_queue) and
        termination is driven by the _STREAM_STOP marker stop_recording puts on
        it (not by polling self.recording): mlx_executor has one worker, so if a
        new recording starts before this session's queue has fully drained, a
        shared self.recording/self._stream_queue would get reassigned to the new
        session out from under this still-running loop.
        """
        import mlx.core as mx

        self._typed_finalized = ""
        self._typed_draft = ""
        batch = bytearray()
        got_stop = False
        # Preset chosen live from the Accuracy menu (see ACCURACY_PRESETS in
        # settings.py for what each knob does and why raising both context and
        # depth is safe now that _apply_stream_result has its shrink guard).
        preset = ACCURACY_PRESETS[self.settings["accuracy_mode"]]
        batch_target_bytes = int(preset["batch_seconds"] * self.rate) * 2

        try:
            with self.model.transcribe_stream(
                context_size=preset["context_size"], depth=preset["depth"]
            ) as stream:
                while True:
                    try:
                        item = stream_queue.get(timeout=0.05)
                        if item is _STREAM_STOP:
                            got_stop = True
                        else:
                            batch += item
                    except queue.Empty:
                        pass

                    stopped = got_stop and stream_queue.empty()
                    if batch and (len(batch) >= batch_target_bytes or stopped):
                        pcm16 = np.frombuffer(bytes(batch), dtype=np.int16)
                        audio_np = pcm16.astype(np.float32) / 32768.0
                        # Per-chunk auto-gain so quiet/whispered speech gets a fair shot
                        # (a quiet enough clip otherwise decodes to no tokens at all).
                        # RMS-based rather than peak-based: a single louder consonant or
                        # a click in an otherwise-quiet window would cap a peak-based gain
                        # far below what the rest of the (actually quiet) window needs.
                        # Target RMS and cap picked for true whispers, not just "quiet
                        # talking" — clipped afterward since a high gain on a window with
                        # one louder moment can otherwise push samples past full scale.
                        rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
                        # The chunk(s) flushed right after you release the key are
                        # disproportionately likely to be trailing silence, breath, or
                        # the click of releasing the key rather than real speech — you've
                        # already signaled "done talking." Auto-gain would otherwise boost
                        # that noise floor into something the model can mistake for a
                        # mumbled extra word (observed: token count growing in exactly
                        # these low-RMS post-release chunks), so this window needs a
                        # meaningfully louder signal before it's treated as worth boosting.
                        silence_threshold = 0.003 if stopped else 5e-4
                        gain_applied = 1.0
                        if rms > silence_threshold:
                            gain_applied = min(0.05 / rms, 30.0)
                            if gain_applied > 1.0:
                                audio_np = np.clip(audio_np * gain_applied, -1.0, 1.0)
                        audio = mx.array(audio_np)
                        stream.add_audio(audio)
                        logger.info(
                            f"chunk: rms={rms:.5f} gain={gain_applied:.1f} "
                            f"finalized={len(stream.finalized_tokens)} draft={len(stream.draft_tokens)}"
                        )
                        self._apply_stream_result(stream)
                        batch = bytearray()

                    if stopped and not batch:
                        break
        except Exception as e:
            logger.error(f"Error during live transcription: {e}")
            self.status_item.title = "Status: Error during live transcription"
        else:
            preview = (self._typed_finalized + self._typed_draft)[:30]
            logger.info(f"Session ended. Typed: {preview!r}")
            self.status_item.title = (
                f"Status: Transcribed: {preview}..." if preview else "Status: No speech detected"
            )
        finally:
            self.title = "🎙️"

    def process_recording(self):
        try:
            self.transcribe_audio()
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            self.status_item.title = "Status: Error during transcription"
        finally:
            self.title = "🎙️"

    def _write_wav_to_buffer(self, frames_bytes: bytes) -> io.BytesIO:
        """Create an in-memory WAV buffer from PCM frames."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(frames_bytes)
        buf.seek(0)
        return buf

    def transcribe_audio(self):
        if not self.frames:
            self.title = "🎙️"
            self.status_item.title = "Status: No audio recorded"
            logger.warning("No audio recorded")
            return

        pcm = b''.join(self.frames)

        # Always use temp file for transcription (model does not support BytesIO)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(pcm)
            result = self.model.transcribe(temp_filename)
        finally:
            try:
                os.unlink(temp_filename)
            except Exception:
                pass

        text = (getattr(result, "text", "") or "").strip()

        if text:
            selected_text = self.text_selector.get_selected_text()

            # NEW: If there is selected text and local Qwen is ready → treat spoken text as instruction
            # (settings check so disabling via the menu takes effect immediately even after
            # the model's already loaded, without needing to unload it)
            if selected_text and self.llm_ready and self.settings["llm_enabled"]:
                try:
                    self.status_item.title = "Status: Editing selection with Qwen..."
                    edited = self.enhance_with_qwen(text, selected_text)
                    if edited:
                        self.text_selector.replace_selected_text(edited)
                        self.status_item.title = f"Status: Edited: {edited[:30]}..."
                    else:
                        # Fallback to inserting raw transcription if model returned nothing
                        self.insert_text(text)
                        self.status_item.title = f"Status: Transcribed: {text[:30]}..."
                except Exception as e:
                    logger.error(f"Qwen edit error: {e}")
                    self.insert_text(text)
                    self.status_item.title = f"Status: Transcribed: {text[:30]}..."
            else:
                # No selection or LLM not ready → normal dictation insert
                self.insert_text(text)
                self.status_item.title = f"Status: Transcribed: {text[:30]}..."
        else:
            logger.warning("No speech detected")
            self.status_item.title = "Status: No speech detected"

    def insert_text(self, text):
        # Minimal logging in hot path
        if self.settings["symbol_words_enabled"]:
            text = _apply_symbol_words(text)
        self.keyboard_controller.type(text)

    # NEW: Local edit using Qwen (MLX)
    def enhance_with_qwen(self, instruction: str, text: str) -> str:
        """
        Use local Qwen (MLX) to apply `instruction` to `text`.
        Returns edited text (no explanations, no wrappers).
        """
        logger.debug(f"Enhancing with Qwen | instruction: {instruction}")
        if not (self.llm_model and self.llm_tokenizer):
            return ""

        system = (
            "You are a local text editor. Apply the instruction precisely. "
            "Output only the final edited text, with no extra formatting, no brackets, no explanations, and no wrappers."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Instruction: {instruction}\nText:\n{text}"},
        ]

        # Try chat template; fallback to a simple prompt if unavailable
        try:
            prompt = self.llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            logger.debug("got prompt from llm tokenizer chat template")
        except Exception:
            prompt = f"{system}\n\nInstruction: {instruction}\nText:\n{text}\n\nEdited:"
            logger.debug("got prompt from simple chat template")
        logger.debug(prompt)
        out = mlx_generate(
            self.llm_model,
            self.llm_tokenizer,
            prompt=prompt,
            max_tokens=self.llm_config["max_tokens"],
            verbose=False,
        )
        logger.debug("Qwen output is " + str(out))
        return self._clean_llm_output(out)

    def _clean_llm_output(self, out: str) -> str:
        """
        Remove any <<<...>>> wrappers or similar from LLM output.
        """
        if not out:
            return ""
        # Remove <<< ... >>> wrappers
        import re
        cleaned = re.sub(r"^\s*<<<(.*?)>>>\s*/?s?\s*$", r"\1", out.strip(), flags=re.DOTALL)
        # Remove any remaining brackets or slashes
        cleaned = cleaned.strip().strip('<>').strip('/s').strip()
        return cleaned

    def handle_shutdown(self, _signal, _frame):
        pass

def main():
    parser = argparse.ArgumentParser(
        description=f"Parakeet Dictation: Speech-to-text and local LLM text editing for macOS.\n\nINSTRUCTIONS:\n\n- After launching, look for the 🎙️ icon in your macOS menu bar.\n- Press and HOLD {_hotkey_display_name(DEFAULT_HOTKEY)} to start dictation (configurable from the menu bar). Release to transcribe.\n- If you select text before dictating, your spoken command will be used as an edit instruction for the selected text (requires local LLM).\n- If hotkeys do not work, check System Settings → Privacy & Security → Accessibility and Input Monitoring.\n- To quit, use the menu bar or press Ctrl+C in the terminal.\n- For more info, see: https://github.com/osadalakmal/parakeet-dictation\n\nOPTIONS:"
    )
    parser.add_argument('--version', action='version', version='parakeet-dictation 0.1.0')
    args = parser.parse_args()

    try:
        WhisperDictationApp().run()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, exiting...")
        os._exit(0)

if __name__ == "__main__":
    main()
