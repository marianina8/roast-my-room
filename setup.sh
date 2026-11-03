#!/usr/bin/env bash
# One-time setup on a Mac: imagesnap, yzma, the llama.cpp libraries and a vision model.
set -euo pipefail
cd "$(dirname "$0")"

LIB="$HOME/yzma/lib"
MODELS="$HOME/models"
HF="https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main"

command -v go >/dev/null || { echo "Install Go first: https://go.dev/dl/"; exit 1; }

if ! command -v imagesnap >/dev/null; then
  echo "==> Installing imagesnap (takes webcam photos)"
  brew install imagesnap
fi

echo "==> Installing the yzma command"
go install github.com/hybridgroup/yzma@v1.31.0
YZMA="$(go env GOPATH)/bin/yzma"

echo "==> Downloading the llama.cpp libraries (Metal) into $LIB"
mkdir -p "$LIB"
"$YZMA" install --lib "$LIB" --processor metal

# The brain and its eyes: Qwen3-VL 4B (the URL pattern yzma's own Makefile
# uses for Qwen3-VL), falling back to the 2B version if the 4B files aren't there.
download() { # repo file
  if [ -f "$MODELS/$2" ]; then
    echo "==> $2 is already downloaded"
  else
    echo "==> Downloading $2"
    "$YZMA" model get -y -u "https://huggingface.co/$1/resolve/main/$2" -o "$MODELS"
  fi
}

mkdir -p "$MODELS"
SIZE=4B
if ! curl -sfIL -o /dev/null "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf"; then
  echo "==> Qwen3-VL 4B not found, using 2B"
  SIZE=2B
fi
REPO="Qwen/Qwen3-VL-$SIZE-Instruct-GGUF"
MODEL="Qwen3VL-$SIZE-Instruct-Q4_K_M.gguf"
PROJ="mmproj-Qwen3VL-$SIZE-Instruct-Q8_0.gguf"
download "$REPO" "$MODEL"
download "$REPO" "$PROJ"

echo "==> Building roast-my-room"
go mod tidy
go build -o roast-my-room .

ls -lh "$MODELS/$MODEL" "$MODELS/$PROJ"
echo
echo "Ready. Run:  ./roast-my-room"
