#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[init] cargo test (adversarial suite)"
cargo test --release

echo "[init] cargo build --release"
cargo build --release

echo "[init] running Rust demo"
./target/release/titanmk demo

echo "[init] validating sample schedule"
./target/release/titanmk validate zig/samples/linear.json --verbose

echo "[init] generating CUDA source (codegen-only smoke test, no nvcc required)"
mkdir -p cuda/generated
./target/release/titanmk gen-cuda cuda/samples/linear.json -o cuda/generated/generated_megakernel.cu --tensor-len 8
echo "[init] CUDA codegen wrote cuda/generated/generated_megakernel.cu + tensor_fields.inc"

if command -v nvcc >/dev/null 2>&1; then
  echo "[init] nvcc found -- building CUDA demo"
  (cd cuda && make build)
else
  echo "[init] nvcc not found -- skipping native CUDA build (use cuda-builder Docker stage on a GPU host)"
fi

if command -v zig >/dev/null 2>&1; then
  echo "[init] zig codegen + build + run"
  cd zig
  mkdir -p generated
  zig build run -- gen samples/linear.json > generated/generated_megakernel.zig
  zig build
  zig build demo
  cd ..
else
  echo "[init] zig not found locally -- skipping native Zig build (Docker stage handles it)"
fi

echo "[init] building docker image titan-megakernel:latest"
docker build -t titan-megakernel:latest .

echo "[init] done. run with: docker run --rm titan-megakernel:latest"
