# TitanMegaKernel

TitanMegaKernel validates compute schedules before code generation and rejects unsafe execution plans before they reach hardware.

[![TitanMegaKernel Proof](https://github.com/juliushill42/TitanMegaKernel/actions/workflows/verify.yml/badge.svg)](https://github.com/juliushill42/TitanMegaKernel/actions/workflows/verify.yml)

### Validated compute schedules with recursive safety checks and CUDA code generation

> **Built by Julius Cameron Hill — TitanU AI LLC**
> Patent Ref: JCH-2026-002 (pending) | Companion: JCH-2026-001
> Status: Phases 1–3 complete

---

## What This Is

AURORA is a system that takes an AI model's execution plan, **mathematically proves it is safe** (no deadlocks, no data races, no crashes — before a single line runs), and then generates working code for **any hardware target** from that single certified proof.

Same proof. CPU or GPU. One certificate that covers both.

---

## Why It Matters

Standard AI runtimes launch one operation at a time — multiply, save, normalize, save, attend, save. Every "save" is a round trip through memory. That's the bottleneck.

The solution is to fuse all those operations into one continuous run and **prove** the fused plan is safe before executing it. Recent research (AutoMegaKernel / AMK) did this — but only for NVIDIA GPUs.

AURORA generalizes it:

| | AMK (RightNow AI) | AURORA (TitanU AI) |
|---|---|---|
| Validator | ✅ Static, certified | ✅ Static, certified + **recursive** |
| CPU target | ❌ | ✅ Zig / AVX2 |
| NVIDIA GPU | ✅ Fused cooperative kernel | ✅ Single-stream CUDA (fused next) |
| AMD / Apple | ❌ | 🔜 Roadmap |
| Hardware lock-in | NVIDIA only | **None** |
| Cross-target proof | ❌ | ✅ SHA-256 fingerprint |

---

## The Core Idea: Recursive Schedules

A Schedule is a DAG (graph) of operations. In AURORA, any node in that graph **can itself be a complete Schedule** — making the IR recursive.

This means:
- Certify one transformer block **once**
- Reuse it 32 times across a model without re-proving it
- The validator handles each level independently, then treats certified sub-schedules as atomic

```
Model Schedule
├── Node 0: Embedding
├── Node 1–32: TransformerBlock (SubSchedule — certified once, reused 32x)
│   ├── Node A: RMSNorm
│   ├── Node B: Attention
│   ├── Node C: MatMul
│   └── Node D: Softmax
└── Node 33: Output projection
```

---

## The Validator: 6 Guarantees Before Anything Runs

```
titanmk validate my_schedule.json
```

Runs 6 checks, recursively, across every nesting level:

1. **Recursive sub-schedule validation** — nested blocks certified independently first
2. **Dangling-wait check** — every dependency reference must resolve to a real node
3. **Cycle detection** — DFS over the wait graph; no deadlocks, guaranteed
4. **Per-compute-unit queue-order** — no two operations on the same unit with an unresolved conflict
5. **Race-freedom** — no two concurrent operations touching the same memory with a write
6. **Certificate + fingerprint** — SHA-256 hash of the canonical schedule, issued on success

If validation fails → **codegen is blocked entirely.** An uncertified schedule never reaches hardware.

---

## Architecture

```
titan-megakernel/
├── src/
│   ├── ir.rs               # Recursive Schedule IR (Node, OpKind, SubSchedule)
│   ├── validator.rs        # 6-check static validator + Certificate
│   ├── adversarial.rs      # Enumerated adversarial test suite + CUDA smoke tests
│   ├── cuda_codegen.rs     # Recursive .cu codegen (gen-cuda subcommand)
│   └── main.rs             # titanmk CLI
├── zig/
│   ├── src/
│   │   ├── schedule.zig    # JSON deserializer mirroring Rust IR
│   │   ├── codegen.zig     # Recursive topo-sort + Zig emission
│   │   ├── titan_kernels.zig  # RMSNorm, Softmax, MatMul, Attention, Elementwise
│   │   ├── demo.zig        # Demo runner for generated megakernel
│   │   └── main.zig        # titanmk-zig CLI
│   ├── build.zig
│   └── Makefile
└── cuda/
    ├── kernels/titan_kernels.cuh   # RMSNorm, Softmax, MatMul/cuBLAS, Attention
    ├── src/main.cu                 # Host harness
    └── Makefile
```

---

## Quickstart

### Prerequisites
- [Rust](https://rustup.rs) (stable)
- [Zig](https://ziglang.org/download/) 0.13.0 (for CPU backend)
- nvcc + CUDA 12.x (for GPU backend, optional)
- Docker (optional, handles everything)

### One-Command Build
```bash
tar xzf titan-megakernel.tar.gz
cd titan-megakernel
./init.sh
```

`init.sh` will:
1. Run the full adversarial test suite (`cargo test`)
2. Build the `titanmk` CLI
3. Run the demo (7 schedule cases, pass/fail report)
4. Validate the sample schedule
5. Generate CUDA source (no GPU needed for this step)
6. Build and run the Zig CPU demo (if Zig is installed)
7. Build and push the Docker image

---

## CLI Reference

```bash
# Validate a schedule (blocks codegen if it fails)
titanmk validate schedule.json --verbose

# Get the SHA-256 fingerprint only
titanmk fingerprint schedule.json

# Run the built-in adversarial demo (7 pass/fail cases)
titanmk demo

# Validate + generate CUDA source
titanmk gen-cuda schedule.json -o generated/megakernel.cu --tensor-len 8
```

```bash
# Generate Zig CPU source
cd zig
zig build run -- gen samples/linear.json > generated/generated_megakernel.zig
zig build demo
```

---

## CPU Backend (Zig / AVX2)

**No GPU required. Runs on any machine.**

```
Schedule JSON
     ↓
titanmk-zig gen
     ↓
run_megakernel() [Zig source]
     ↓
titan_kernels.zig [RMSNorm · Softmax · MatMul · Attention · Elementwise]
     ↓
Native binary
```

**Integration point:** `zig/src/titan_kernels.zig` currently ships portable scalar reference implementations. Drop your AVX2 Titan Kernels implementations into these function bodies — signatures are the fixed contract, no codegen changes required.

---

## GPU Backend (CUDA)

```bash
# Tiled shared-memory GEMM
cd cuda && make run

# cuBLAS path (for benchmarking vs AMK's L4/L40S/5090 numbers)
cd cuda && make USE_CUBLAS=1 run

# Docker (no local CUDA install needed)
docker build --target cuda-builder -t titan-megakernel-cuda .
docker run --rm --gpus all titan-megakernel-cuda
```

**Kernel library** (`cuda/kernels/titan_kernels.cuh`):

| Kernel | Implementation |
|---|---|
| RMSNorm | One block per row, shared-memory tree reduction |
| Softmax | Two-pass shared-memory (max then sum), numerically stable |
| MatMul | 16×16 tiled shared-memory GEMM, optional cuBLAS via `-DTITAN_USE_CUBLAS` |
| Attention | One block per (head, query), in-block softmax, `rsqrtf` scale |
| Elementwise | Single fused kernel: id / relu / gelu / silu |
| Barrier | `cudaStreamSynchronize` |

> **Current GPU scope:** sequential launches on one CUDA stream — no host round-trips between ops. Single cooperative-groups megakernel (one `__global__` launch for the entire forward pass) is the next milestone and requires GPU hardware for occupancy tuning.

---

## Cross-Target Equivalence: The Fingerprint

```bash
titanmk fingerprint schedule.json
# → a7f3c91b2e048d...
```

Because the SHA-256 fingerprint is computed from the canonical schedule **before any codegen**, two binaries — one AVX2, one CUDA — built from the same fingerprint are provably executing the identical validated dependency graph.

The proof is portable. The machine code is not. That distinction is the moat.

---

## Patent Claims Summary (JCH-2026-002)

1. **Recursive Schedule IR with Self-Similar Certification** — nested schedules certified independently; parent level treats them as atomic
2. **Hardware-Agnostic Codegen from a Single Certified IR** — one validator, one certificate, multiple target backends
3. **Cryptographic Schedule Fingerprinting as Cross-Target Equivalence Proof** — SHA-256 issued pre-codegen, valid across all targets
4. **Per-Compute-Unit Queue-Order Certification (Hardware-Neutral)** — SmId abstraction generalizes CUDA SM ordering to any pinned-unit architecture

---

## Roadmap

| Item | Status |
|---|---|
| Recursive IR + 6-check validator | ✅ Complete |
| Adversarial test suite | ✅ Complete |
| Zig CPU backend (scalar ref kernels) | ✅ Complete |
| AVX2 Titan Kernels integration | 🔧 Swap-in ready |
| CUDA sequential single-stream backend | ✅ Complete |
| Fused cooperative-groups CUDA megakernel | 🔜 Next (requires GPU hardware) |
| ROCm/HIP backend | 🔜 Architecturally unblocked |
| Metal backend | 🔜 Architecturally unblocked |
| Per-tensor shape propagation in codegen | 🔧 Near-term hardening |
| Property-based adversarial expansion | 🔧 Near-term hardening |

---

## Built By

**Julius Cameron Hill**
Founder & Sole Developer — TitanU AI LLC
[titanuai.com](https://titanuai.com)

Self-taught. No CS degree. Started October 2025.
This is what sovereign infrastructure looks like.

---

*© 2026 TitanU AI LLC. All rights reserved. Patent pending JCH-2026-002.*
