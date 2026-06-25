#!/usr/bin/env bash
# Install Ollama (sudoless if needed) and pull the local coding models optarena
# drives as agents. Works on Linux, WSL, and macOS.
#
#   scripts/install_ollama.sh                         # install + pull the defaults
#   scripts/install_ollama.sh qwen2.5-coder:32b       # + extra model(s)
#   OPTARENA_OLLAMA_MODELS="qwen2.5-coder:7b" scripts/install_ollama.sh   # override list
#   OPTARENA_OLLAMA_PREFIX=$HOME/.local scripts/install_ollama.sh         # install prefix
#
# The canonical model is qwen2.5-coder:7b (chat/edit); qwen2.5-coder:1.5b is the
# fast autocomplete model. The optarena agent talks to the running server over
# HTTP -- no Python deps:  python -m optarena.cli agent --agent ollama ...
set -euo pipefail

# --- defined model list (canonical first) ----------------------------------
# qwen2.5-coder:7b   -> chat / edit / multi-file agent work (Aider, optarena agent)
# qwen2.5-coder:1.5b -> snappy tab-autocomplete (Continue.dev), tiny + CPU-friendly
DEFAULT_MODELS=("qwen2.5-coder:7b" "qwen2.5-coder:1.5b")
# Other popular local coders (pass as args to also pull):
#   qwen2.5-coder:32b  deepseek-coder-v2:16b  codellama:13b  starcoder2:7b

PREFIX="${OPTARENA_OLLAMA_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_HOST

have() { command -v "$1" >/dev/null 2>&1; }

ensure_ollama() {
  if have ollama; then echo "ollama found: $(command -v ollama)"; return; fi
  if [ -x "$BIN_DIR/ollama" ]; then
    export PATH="$BIN_DIR:$PATH"; echo "ollama found: $BIN_DIR/ollama"; return
  fi
  echo "ollama not found -- installing WITHOUT sudo into $PREFIX ..."
  mkdir -p "$BIN_DIR"
  local os arch tmp
  os="$(uname -s)"; arch="$(uname -m)"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  case "$os" in
    Linux)
      case "$arch" in
        x86_64|amd64) arch=amd64 ;;
        aarch64|arm64) arch=arm64 ;;
        *) echo "unsupported arch: $arch" >&2; exit 1 ;;
      esac
      # The official release ships .tar.zst (newer) with a .tgz fallback (older);
      # both unpack to bin/ollama (+ lib/ollama) under the prefix -- no sudo.
      local base="https://ollama.com/download/ollama-linux-${arch}"
      if curl -fsSL "${base}.tar.zst" -o "$tmp/o.tar.zst" 2>/dev/null; then
        _extract_zst "$tmp/o.tar.zst" "$PREFIX"
      else
        echo "  .tar.zst unavailable; falling back to .tgz"
        curl -fsSL "${base}.tgz" -o "$tmp/o.tgz"
        tar -xzf "$tmp/o.tgz" -C "$PREFIX"
      fi
      ;;
    Darwin)
      if have brew; then
        echo "  using Homebrew (sudoless): brew install ollama"
        brew install ollama
        echo "ollama found: $(command -v ollama)"; return
      fi
      # No brew: unpack the app bundle and lift out its CLI binary.
      echo "  downloading Ollama-darwin.zip (no brew found)"
      curl -fsSL "https://ollama.com/download/Ollama-darwin.zip" -o "$tmp/o.zip"
      unzip -q "$tmp/o.zip" -d "$tmp"
      local cli
      cli="$(find "$tmp" -type f -name ollama -perm -u+x | head -1)"
      [ -n "$cli" ] || { echo "could not find the ollama CLI in the app bundle" >&2; exit 1; }
      cp "$cli" "$BIN_DIR/ollama"; chmod +x "$BIN_DIR/ollama"
      ;;
    *) echo "unsupported OS: $os (Linux/WSL/macOS only)" >&2; exit 1 ;;
  esac
  export PATH="$BIN_DIR:$PATH"
  have ollama || { echo "install failed: ollama not on PATH" >&2; exit 1; }
  echo "installed: $(command -v ollama)"
  echo "NOTE: add '$BIN_DIR' to PATH in your shell rc to keep ollama available."
}

# Decompress a .tar.zst with whatever the host provides (GNU tar --zstd, or the
# zstd/unzstd CLI piped into tar) so we don't hard-require a particular tool.
_extract_zst() {
  local archive="$1" dest="$2"
  if tar --help 2>&1 | grep -q -- '--zstd'; then tar -x --zstd -f "$archive" -C "$dest"
  elif have zstd;   then zstd -dc  "$archive" | tar -xf - -C "$dest"
  elif have unzstd; then unzstd -c "$archive" | tar -xf - -C "$dest"
  else echo "need zstd (or GNU tar --zstd) to unpack $archive" >&2; exit 1; fi
}

ensure_server() {
  if curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo "ollama server already running at $OLLAMA_HOST"; return
  fi
  echo "starting 'ollama serve' in the background ..."
  nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
  local i
  for i in $(seq 1 30); do
    curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && { echo "server up"; return; }
    sleep 1
  done
  echo "server did not become ready -- see /tmp/ollama_serve.log" >&2; exit 1
}

main() {
  ensure_ollama
  ensure_server
  local models
  if [ -n "${OPTARENA_OLLAMA_MODELS:-}" ]; then
    read -r -a models <<< "$OPTARENA_OLLAMA_MODELS"
  else
    models=("${DEFAULT_MODELS[@]}")
  fi
  models+=("$@")
  echo "pulling: ${models[*]}"
  local m
  for m in "${models[@]}"; do
    echo "=== ollama pull $m ==="
    ollama pull "$m"
  done
  cat <<EOF

Done. Models are served at $OLLAMA_HOST.
  - optarena benchmark agent : python -m optarena.cli agent --agent ollama --kernels gemm --languages c
  - terminal coding agent   : aider --model ollama/qwen2.5-coder:7b
See docs/local_coding_agents.md.
EOF
}
main "$@"
