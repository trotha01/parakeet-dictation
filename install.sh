#!/bin/bash
# Install Parakeet Dictation: https://github.com/trotha01/parakeet-dictation
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/trotha01/parakeet-dictation/main/install.sh | bash
#
# Safe to re-run: every step below is idempotent.

set -euo pipefail

REPO_URL="https://github.com/trotha01/parakeet-dictation.git"
REPO_BRANCH="main"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$1" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "Parakeet Dictation only runs on macOS."
[ "$(uname -m)" = "arm64" ] || die "Parakeet Dictation needs Apple Silicon (MLX doesn't run on Intel Macs)."

step "Checking for Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  die "Homebrew is required for portaudio. Install it from https://brew.sh, then re-run this script."
fi
echo "found: $(brew --version | head -1)"

step "Installing portaudio (needed for microphone access)"
brew list portaudio >/dev/null 2>&1 && echo "already installed" || brew install portaudio

step "Checking for uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "not found, installing via the official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer adds uv to a shell profile for *future* shells; make it available in this one too.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — open a new terminal and re-run this script."
else
  echo "found: $(uv --version)"
fi

step "Installing parakeet-dictation"
uv tool install --force "git+${REPO_URL}@${REPO_BRANCH}"

UV_BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
case ":$PATH:" in
  *":$UV_BIN_DIR:"*) ;;
  *) warn "$UV_BIN_DIR isn't on your PATH yet. Add it to your shell profile (e.g. ~/.zshrc):
    export PATH=\"$UV_BIN_DIR:\$PATH\"
  Then open a new terminal before running parakeet-dictation." ;;
esac

step "macOS permissions"
echo "Parakeet Dictation needs three permissions granted to whatever app you launch it"
echo "from (this terminal, if you're reading this here) — opening the relevant System"
echo "Settings panes now. In each one, add/enable your terminal app:"
echo "  - Accessibility     (to type at your cursor in other apps)"
echo "  - Input Monitoring  (to detect the right-Option push-to-talk key globally)"
echo "  - Microphone        (to actually hear you — without this it records silence"
echo "                        with no error, see the README's Troubleshooting section)"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || true
open "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent" 2>/dev/null || true
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone" 2>/dev/null || true

step "Done"
echo "After granting those three permissions, launch it with:"
echo "  parakeet-dictation"
echo ""
echo "Then hold right Option, talk, and release. Full docs (env vars, troubleshooting):"
echo "  https://github.com/trotha01/parakeet-dictation/blob/${REPO_BRANCH}/README.md"
