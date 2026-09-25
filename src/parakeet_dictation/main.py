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

class WhisperDictationApp(rumps.App):
    def __init__(self):
        super(WhisperDictationApp, self).__init__("🎙️", quit_button=rumps.MenuItem("Quit"))
        self.status_item = rumps.MenuItem("Status: Ready")
        self.recording_menu_item = rumps.MenuItem("Start Recording")
        self.menu = [self.recording_menu_item, None, self.status_item]

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
        self.llm_config = {
            "model_id": os.getenv("PARAKEET_LLM_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
            "max_tokens": int(os.getenv("PARAKEET_LLM_MAX_TOKENS", "192")),
            "temperature": float(os.getenv("PARAKEET_LLM_TEMP", "0.2")),
            "top_p": float(os.getenv("PARAKEET_LLM_TOP_P", "0.9")),
        }
        # The speak-an-edit rewrite loads Qwen2.5-1.5B eagerly at startup and holds it
        # resident for a feature that is only used deliberately. Off unless asked for.
        if os.environ.get("PARAKEET_ENABLE_LLM") == "1":
            self.load_llm_thread = threading.Thread(target=self.load_llm, daemon=True)
            self.load_llm_thread.start()
        else:
            logger.info("LLM rewrite disabled (set PARAKEET_ENABLE_LLM=1 to enable)")

        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 512  # smaller chunk -> snappier stop

        # Hotkey state
        self.is_recording_with_hotkey = False

        # Set up global hotkeys (Ctrl+Alt+A) and release listener
        self.setup_global_monitor()

        logger.info("Started WhisperDictation app. Look for 🎙️ in your menu bar.")
        logger.info("Press and HOLD Ctrl + Alt + A to record. Release to transcribe.")
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
        except Exception as e:
            logger.error(f"Failed to load Qwen MLX model: {e}")
            self.llm_ready = False

    # ---------------------------
    # Global hotkey + release monitor
    # ---------------------------
    def setup_global_monitor(self):
        self.key_monitor_thread = threading.Thread(target=self.monitor_keys, daemon=True)
        self.key_monitor_thread.start()

    def monitor_keys(self):
        """
        Hold RIGHT Option to record; release to transcribe.

        A bare modifier cannot be expressed as a GlobalHotKeys chord, so this
        uses a plain Listener on one key. LEFT Option is deliberately ignored,
        so every normal Option shortcut keeps working.
        """
        TRIGGER = keyboard.Key.alt_r

        def on_press(key):
            if key == TRIGGER and not self.recording and not self.is_recording_with_hotkey:
                self.is_recording_with_hotkey = True
                logger.info("STARTING recording via right-Option")
                self.start_recording()

        def on_release(key):
            if key == TRIGGER and self.is_recording_with_hotkey and self.recording:
                logger.info("STOPPING recording via right-Option release")
                self.is_recording_with_hotkey = False
                self.stop_recording()

        logger.info("Global hotkey listener: hold RIGHT Option to record")
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
        except Exception as e:
            logger.error(f"Error with keyboard listeners: {e}")
            logger.error("Please check Accessibility/Input Monitoring permissions in System Settings.")

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
        self.title = "🎙️ (Recording)"
        self.status_item.title = "Status: Recording..."
        logger.info("Recording started. Speak now...")

        # Text selected before you start talking means "speak an edit instruction for
        # this selection" (Qwen path below) — that stays batch-transcribed on release.
        # Otherwise, type live as you talk via the MLX streaming decoder.
        self._live_mode = not bool(self.text_selector.get_selected_text())
        if self._live_mode:
            self._stream_queue = queue.Queue()

        # Use a callback stream for near-instant stop
        self.recording_thread = threading.Thread(target=self._record_audio_callback_loop, daemon=True)
        self.recording_thread.start()

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

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()

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
        # 0.5s batches: each add_audio call costs ~300ms of fixed overhead almost
        # regardless of chunk size or depth (measured), so small batches waste most
        # of their time on that overhead instead of audio. 0.5s keeps real-time
        # margin (~0.67x) comfortable while still feeling live.
        batch_target_bytes = int(0.5 * self.rate) * 2  # ~0.5s of int16 mono PCM

        try:
            # right_context and depth both cost about the same to run (the
            # ~300ms/call overhead above dominates either way, measured), so
            # there's no real reason not to raise them for accuracy — EXCEPT
            # that parakeet_mlx computes how much audio must arrive before
            # anything finalizes as drop_size = right_context * depth, not
            # either one alone. A first pass at (right=32, depth=24) — maxing
            # depth out since it looked free — gave drop_size = 768 frames,
            # ~61 seconds: nothing finalizes during any normal dictation, so
            # the whole utterance stays in the volatile draft state the
            # entire time, fully exposed to _apply_stream_result's shrink
            # guard above kicking in right at release. Kept right_context
            # (drives the actual per-frame local-attention lookahead, i.e.
            # real accuracy) and dropped depth back down, landing on a
            # drop_size of ~5s — long enough to rarely matter for a typical
            # dictation length, short enough that longer utterances do get
            # real, permanently-locked-in finalized text along the way.
            with self.model.transcribe_stream(context_size=(256, 16), depth=4) as stream:
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
                        # Simple per-chunk auto-gain: a whisper (or just talking quietly)
                        # can sit at a tiny fraction of the mic's usable range, and the
                        # model may register that as near-silence rather than speech (a
                        # quiet enough clip decodes to no tokens at all). Boost toward a
                        # target peak so quiet chunks get a fair shot; capped so we don't
                        # blow up an actually-silent chunk's noise floor.
                        peak = float(np.abs(audio_np).max())
                        if peak > 1e-4:
                            gain = min(0.7 / peak, 8.0)
                            if gain > 1.0:
                                audio_np = audio_np * gain
                        audio = mx.array(audio_np)
                        stream.add_audio(audio)
                        self._apply_stream_result(stream)
                        batch = bytearray()

                    if stopped and not batch:
                        break
        except Exception as e:
            logger.error(f"Error during live transcription: {e}")
            self.status_item.title = "Status: Error during live transcription"
        else:
            preview = (self._typed_finalized + self._typed_draft)[:30]
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
            if selected_text and self.llm_ready:
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
        description="Parakeet Dictation: Speech-to-text and local LLM text editing for macOS.\n\nINSTRUCTIONS:\n\n- After launching, look for the 🎙️ icon in your macOS menu bar.\n- Press and HOLD Ctrl + Alt + A to start dictation. Release to transcribe.\n- If you select text before dictating, your spoken command will be used as an edit instruction for the selected text (requires local LLM).\n- If hotkeys do not work, check System Settings → Privacy & Security → Accessibility and Input Monitoring.\n- To quit, use the menu bar or press Ctrl+C in the terminal.\n- For more info, see: https://github.com/osadalakmal/parakeet-dictation\n\nOPTIONS:"
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
