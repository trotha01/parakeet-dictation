# Parakeet Dictation (macOS)

Local, fast, privacy-friendly dictation for macOS using NVIDIA Parakeet (MLX on Apple Silicon) with a push-to-talk hotkey. Words appear as you talk, not just after you let go.  
Bonus: speak commands to **rewrite selected text** via Qwen (opt-in).

---

## Table of Contents

- [Why this project?](#why-this-project)
- [Features](#features)
- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Push-to-talk dictation](#push-to-talk-dictation)
  - [Voice-driven text editing (Qwen via MLX)](#voice-driven-text-editing-qwen-via-mlx)
  - [Menu bar controls](#menu-bar-controls)
- [Permissions (macOS)](#permissions-macos)
- [Run in the background](#run-in-the-background)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [How this fork differs from upstream](#how-this-fork-differs-from-upstream)
- [Credits](#credits)
- [License](#license)

---

## Why this project?

Parakeet Dictation gives you **on-device** speech-to-text on macOS with a **single push-to-talk key**. It’s built to be:

- **Private**: Audio is processed locally on your Mac.
- **Fast**: Parakeet models are optimized and run great on Apple Silicon via MLX.
- **Practical**: Dictate into *any* app, or select text and **say how to transform it** (“make this more professional”, “translate to Spanish”, etc.) — the app rewrites it via a local Qwen2.5 model and pastes it in place.

---

## Features

- 🖥️ **Menu bar** app (stays out of your way)
- 🎙️ **Push-to-talk**: hold **right Option** to record, release to stop — left Option is left alone so its normal shortcuts still work
- ⚡ **Live streaming dictation**: words are typed as you talk (via `parakeet-mlx`'s streaming decoder), not only after you release the key
- ⚡ **Local ASR** with **NVIDIA Parakeet** (Apple Silicon via MLX)
- ⌨️ **Types directly at the cursor** in the foreground app
- ✨ **Voice-driven text editing** (optional, off by default — see `PARAKEET_ENABLE_LLM`): when text is selected, your speech is treated as an instruction and the selection is replaced with the result (via local Qwen2.5 on MLX)
- ✅ Clear **recording status** via the menu bar icon
- 🧰 Simple **background mode** (no UI) for power users

---

## Demo

TBD

---

## Requirements

- **Apple Silicon Mac** (MLX requires it)
- **Python 3.12** (pinned via `requires-python` in `pyproject.toml`)
- **Microphone**
- **Accessibility + Input Monitoring permission** (for the global push-to-talk key and typing into other apps)
- **PortAudio** (for PyAudio)
- [**uv**](https://docs.astral.sh/uv/) (recommended — manages its own isolated Python and venv, no manual `venv`/`activate` needed)

---

## Installation

### Option A: one-line install (recommended)

Installs Homebrew's `portaudio`, `uv` if you don't have it, and the app itself, then opens the three permission panes you'll need to approve:

```bash
curl -fsSL https://raw.githubusercontent.com/trotha01/parakeet-dictation/main/install.sh | bash
```

Safe to re-run — every step is idempotent. Requires [Homebrew](https://brew.sh) already installed (the script checks and tells you if it isn't).

### Option B: manual install

```bash
brew install portaudio
uv tool install "git+https://github.com/trotha01/parakeet-dictation.git"
```

### Option C: hack on the source

What this fork's own development uses:

```bash
brew install portaudio
git clone https://github.com/trotha01/parakeet-dictation.git
cd parakeet-dictation
uv tool install --editable .
```

With `--editable`, edits to `src/parakeet_dictation/main.py` take effect on the next launch — no reinstall needed.

All three options install a `parakeet-dictation` executable (check with `which parakeet-dictation`).

---

## Configuration

All optional, set as environment variables before launching. `PARAKEET_ENABLE_LLM` and `PARAKEET_LOG` are also available live from the menu bar (see [Menu bar controls](#menu-bar-controls)) — the menu is where those two actually persist across relaunches; the env vars just seed that setting for whichever launch sets them.

| Variable | Default | Purpose |
|---|---|---|
| `PARAKEET_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | ASR model to use, any `parakeet-mlx`-compatible MLX Hub repo |
| `PARAKEET_ENABLE_LLM` | unset (off) | Set to `1` to load Qwen2.5-1.5B and enable voice-driven text editing. Off by default: loading it downloads/holds a second model in memory for a feature most launches won't use |
| `PARAKEET_LLM_MODEL` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | Edit-instruction model, only relevant with `PARAKEET_ENABLE_LLM=1` |
| `PARAKEET_LLM_MAX_TOKENS` | `192` | Max tokens for an edit rewrite |
| `PARAKEET_LLM_TEMP` | `0.2` | Edit rewrite temperature |
| `PARAKEET_LLM_TOP_P` | `0.9` | Edit rewrite top-p |
| `PARAKEET_LOG` | `warning` | Set to `info` or `debug` for verbose logging — `info` includes per-chunk audio level (RMS), auto-gain applied, and live token counts, useful for diagnosing why a session came out blank or garbled |

---

## Usage

### Push-to-talk dictation

1. Launch the app (see Development or Background sections below).
2. Click into any text field, then hold **right Option** to start recording. (Left Option is deliberately ignored, so its usual shortcuts — e.g. option-drag, special characters — keep working everywhere else.)
3. Speak normally. Words are typed as you go, not only at the end.
4. Release right Option to stop. Any words still being refined by the decoder get one last reconciliation pass.

### Voice-driven text editing (Qwen via MLX)

Requires `PARAKEET_ENABLE_LLM=1` (see [Configuration](#configuration)) — off by default.

1. Select text in any app.
2. Hold right Option and speak an instruction, e.g.:
  - “Fix the grammar”
  - “Summarize this”
  - “Translate to Spanish”
3. Release right Option. The selected text is replaced with the edited version. (This path is batch, not live-typed — an edit instruction needs to be heard in full before it can be applied.)

When no text is selected, your speech is always treated as plain dictation, whether or not the LLM edit feature is enabled.

### Menu bar controls

- **Start/Stop Recording** — toggles recording (same as holding right Option)
- **Accuracy** — Fast / Balanced / Accurate, the presets from [Configuration](#configuration); takes effect on your next recording
- **Enable Text Editing (Qwen)** — same as `PARAKEET_ENABLE_LLM`, but live: toggling it on loads Qwen in the background on first use rather than requiring a restart. Shows "(downloads ~1GB)" until the model is actually cached
- **Verbose Logging** — same as `PARAKEET_LOG=info`, toggled live
- **Status: ...** — not clickable, just shows current state
- **Quit** — exits the app

All three settings persist across relaunches (`~/Library/Application Support/parakeet-dictation/settings.json`). The `PARAKEET_*` env vars still work too — they seed that setting for the launch they're set on, but the menu is what's saved.

---

## Permissions (macOS)

Grant all three to whatever app actually launches the process — if you run it from a terminal, that's your terminal app (iTerm, Terminal.app, etc.), not `parakeet-dictation` itself:

- **Accessibility**: System Settings → Privacy & Security → Accessibility → allow your terminal app
- **Input Monitoring**: System Settings → Privacy & Security → Input Monitoring → allow your terminal app — without this, the global right-Option listener silently never fires
- **Microphone**: System Settings → Privacy & Security → Microphone → allow your terminal app — without this, recording *looks* like it's working (the menu bar shows "Recording...", no error appears) but every session captures silence and ends with "No speech detected"

After granting any of these, fully quit and relaunch the app — permission changes don't apply to an already-running process.

---

## Run in the background

```bash
nohup parakeet-dictation >/tmp/parakeet-dictation.log 2>&1 & disown
```

Stop it later:

```bash
pkill -f parakeet-dictation
```

---

## Troubleshooting

- No audio at all: ensure `portaudio` is installed
- Hotkey does nothing: check Accessibility + Input Monitoring permissions, and that you fully relaunched after granting them
- Nothing types, but no error shown either: almost always missing Microphone permission (see [Permissions](#permissions-macos)) — set `PARAKEET_LOG=info` and try again; a session that captured real speech logs its audio level (`chunk: rms=...`) per chunk, so `rms=0.0` on every line confirms the mic isn't reaching the app
- Launch hangs indefinitely with no log output past the startup warnings: usually a stalled Hugging Face Hub network check, not a real hang — killing and relaunching with `HF_HUB_OFFLINE=1` (once the model is already cached locally) skips it
- High CPU / slow first response: the first recording after launch warms up the model; later ones are faster
- An extra word appears at the end of some dictations: expected occasionally — the chunk(s) right after you release the key are more likely to be trailing breath/noise than speech, and very quiet audio there is deliberately not auto-gained as aggressively as mid-utterance for this reason (see the comments around `silence_threshold` in `main.py` if tuning further)

---

## Development

Install editable (see [Installation](#installation)), then edit `src/parakeet_dictation/main.py` directly — changes take effect on the next launch, no reinstall step:

```bash
pkill -f parakeet-dictation
PARAKEET_LOG=debug parakeet-dictation
```

- Menu bar UI via `rumps`
- Hotkey via `pynput`
- Audio capture via `pyaudio`
- Streaming ASR via `parakeet-mlx`'s `transcribe_stream` (see `_stream_feed_loop` / `_apply_stream_result` in `main.py`)
- AI powered edits via Qwen2.5 on MLX (`mlx-lm`)

**A note on threading with MLX**: `load_model` and every transcription call run on one persistent `concurrent.futures.ThreadPoolExecutor(max_workers=1)`, never on ad-hoc `threading.Thread`s. MLX ties lazily-evaluated arrays to the stream of the thread that created them — loading the model on one thread and transcribing on a fresh thread each time raises `RuntimeError: There is no Stream(cpu, 1) in current thread.` If you add a new code path that calls into the model, route it through `self.mlx_executor`, not a new thread.

---

## Roadmap

- ~~Streaming/partial results~~ — done
- ~~Preferences UI~~ — done, as menu bar items rather than a separate window (see [Menu bar controls](#menu-bar-controls)) — a real window is still possible later if the menu outgrows this
- ~~Configurable streaming latency/accuracy tradeoff~~ — done, the Accuracy menu
- macOS app packaging (Non techie folks can use this as well)
- Real VAD instead of an RMS-threshold heuristic for the post-release trailing-audio cutoff

---

## FAQ

**Does dictation send audio to the cloud?** No. Local only.
**What languages are supported?** English. That's the only one I know :). Feel free to try other languages parkeet supports.

---

## How this fork differs from upstream

This is a fork of [osadalakmal/parakeet-dictation](https://github.com/osadalakmal/parakeet-dictation), diverged over one evening of active development. Substantive changes from upstream `main`:

- **Hotkey**: upstream used a `Ctrl+Alt+A` chord (`keyboard.GlobalHotKeys`, plus a separate listener that stopped on releasing either modifier). This fork holds **right Option** alone — left Option is deliberately left alone so its usual shortcuts (option-drag, special characters) keep working. See [Push-to-talk dictation](#push-to-talk-dictation).
- **Live streaming dictation**: upstream only transcribed once, on release (one batch call). Here, words are typed as you talk via `parakeet-mlx`'s streaming decoder (`transcribe_stream`) — see `_stream_feed_loop` / `_apply_stream_result` in `main.py`.
- **Menu bar settings panel**: added an Accuracy submenu (Fast/Balanced/Accurate presets trading latency for context/depth), an Enable Text Editing toggle, and Verbose Logging — all persisted to `~/Library/Application Support/parakeet-dictation/settings.json` (see `settings.py`). Upstream's menu was just Start/Stop Recording and a status line.
- **Text editing (Qwen) is opt-in**: upstream always loaded the Qwen2.5-1.5B MLX model at startup. Here it's off by default — enable via the "Enable Text Editing (Qwen)" menu item or `PARAKEET_ENABLE_LLM=1`, so a launch that doesn't use it doesn't pay to load/hold it.
- **Default ASR model**: `parakeet-tdt-0.6b-v2` → `parakeet-tdt-0.6b-v3`, and now configurable via `PARAKEET_MODEL` (hardcoded upstream).
- **MLX threading fix**: model loading and every transcription now run on one persistent worker thread (`concurrent.futures.ThreadPoolExecutor(max_workers=1)`) instead of ad-hoc `threading.Thread`s. MLX ties lazily-evaluated arrays to the stream of the thread that created them, so crossing threads — as upstream's ad-hoc threads did — could crash with `RuntimeError: There is no Stream(cpu, 1) in current thread.` (see [Development](#development)).
- **Auto-gain**: quiet/whispered speech gets an RMS-based gain boost per audio chunk before decoding; upstream fed raw levels straight to the model, so quiet audio could decode to no tokens at all.
- **`install.sh`**: a one-line installer (Homebrew `portaudio`, `uv` if missing, the tool itself, then opens the three permission panes) — upstream requires manually cloning and building a venv from `requirements.txt`.
- **README rewritten**: installation moved from a `uv venv` + `requirements.txt` workflow to `uv tool install` (see [Installation](#installation)). Most sections below describe this fork's current behavior, not upstream's.

---

## Credits

- Parakeet MLX (NVIDIA Parakeet on Apple Silicon)
- Originally forked from a [Whisper-based dictation app](https://github.com/ashwin-pc/whisper-dictation)

---

## License

MIT
