# Stage 1: Rust validator
FROM rust:1.79-slim AS rust-builder
WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release --locked || cargo build --release

# Stage 2: Zig codegen + demo
FROM debian:bookworm-slim AS zig-builder
RUN apt-get update && apt-get install -y curl xz-utils && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ -C /opt && \
    ln -s /opt/zig-linux-x86_64-0.13.0/zig /usr/local/bin/zig
WORKDIR /build
COPY --from=rust-builder /build/target/release/titanmk /usr/local/bin/titanmk
COPY zig ./zig
WORKDIR /build/zig
RUN titanmk validate samples/linear.json --verbose
RUN mkdir -p generated && zig build run -- gen samples/linear.json > generated/generated_megakernel.zig
RUN zig build

# Stage 3: runtime
FROM gcr.io/distroless/cc-debian12
COPY --from=rust-builder /build/target/release/titanmk /usr/local/bin/titanmk
COPY --from=zig-builder /build/zig/zig-out/bin/titan-megakernel-demo /usr/local/bin/titan-megakernel-demo
COPY --from=zig-builder /build/zig/zig-out/bin/titanmk-zig /usr/local/bin/titanmk-zig
ENTRYPOINT ["/usr/local/bin/titan-megakernel-demo"]


# ---------------------------------------------------------------------
# Optional Stage 4: CUDA backend (requires nvidia/cuda base + GPU at
# runtime). Not part of the default build target -- build explicitly:
#   docker build --target cuda-builder -t titan-megakernel-cuda .
#   docker run --rm --gpus all titan-megakernel-cuda
# ---------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cuda-builder
COPY --from=rust-builder /build/target/release/titanmk /usr/local/bin/titanmk
WORKDIR /build/cuda
COPY cuda ./
RUN mkdir -p generated && \
    titanmk gen-cuda samples/linear.json -o generated/generated_megakernel.cu --tensor-len 8 && \
    nvcc -O3 -std=c++17 -I kernels -I generated src/main.cu -o generated/titan-megakernel-cuda-demo
ENTRYPOINT ["/build/cuda/generated/titan-megakernel-cuda-demo"]
