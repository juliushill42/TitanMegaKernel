// titan_kernels.cuh
//
// Device kernels for the four primitives (RMSNorm, Softmax, MatMul,
// Attention) plus elementwise/barrier, mirroring zig/src/titan_kernels.zig.
// These are the GPU-side implementations called by the generated
// megakernel dispatch (generated/generated_megakernel.cu).
//
// MatMul defaults to a tiled shared-memory kernel; if TITAN_USE_CUBLAS
// is defined, matmul() instead calls cuBLAS (recommended for parity
// benchmarks against AMK's L4/L40S/5090 numbers).

#pragma once
#include <cuda_runtime.h>
#include <math.h>

#ifdef TITAN_USE_CUBLAS
#include <cublas_v2.h>
#endif

#define TITAN_CUDA_CHECK(call)                                                \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------
// RMSNorm: out[i] = in[i] / sqrt(mean(in^2) + eps)
// One block per row, blockDim.x threads cooperatively reduce.
// ---------------------------------------------------------------------
__global__ void titan_rmsnorm_kernel(float* __restrict__ out,
                                      const float* __restrict__ in,
                                      uint64_t dim, float eps) {
    extern __shared__ float sdata[];
    uint64_t row = blockIdx.x;
    const float* row_in = in + row * dim;
    float* row_out = out + row * dim;

    float local_sum = 0.0f;
    for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = row_in[i];
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)dim + eps);
    for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        row_out[i] = row_in[i] * scale;
    }
}

inline void titan_rmsnorm(float* out, const float* in, uint64_t rows, uint64_t dim,
                           cudaStream_t stream = 0) {
    int threads = 256;
    size_t shmem = threads * sizeof(float);
    titan_rmsnorm_kernel<<<(unsigned)rows, threads, shmem, stream>>>(out, in, dim, 1e-6f);
}

// ---------------------------------------------------------------------
// Softmax: numerically stable, one block per row.
// ---------------------------------------------------------------------
__global__ void titan_softmax_kernel(float* __restrict__ out,
                                      const float* __restrict__ in,
                                      uint64_t dim) {
    extern __shared__ float sdata[];
    uint64_t row = blockIdx.x;
    const float* row_in = in + row * dim;
    float* row_out = out + row * dim;

    float local_max = -INFINITY;
    for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float e = __expf(row_in[i] - max_val);
        row_out[i] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sdata[0];

    for (uint64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        row_out[i] /= sum;
    }
}

inline void titan_softmax(float* out, const float* in, uint64_t rows, uint64_t dim,
                           cudaStream_t stream = 0) {
    int threads = 256;
    size_t shmem = threads * sizeof(float);
    titan_softmax_kernel<<<(unsigned)rows, threads, shmem, stream>>>(out, in, dim);
}

// ---------------------------------------------------------------------
// MatMul: tiled shared-memory GEMM, row-major A (m x k) * B (k x n) -> out (m x n)
// ---------------------------------------------------------------------
#define TITAN_TILE 16

__global__ void titan_matmul_kernel(float* __restrict__ out,
                                     const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     uint64_t m, uint64_t n, uint64_t k) {
    __shared__ float tile_a[TITAN_TILE][TITAN_TILE];
    __shared__ float tile_b[TITAN_TILE][TITAN_TILE];

    uint64_t row = blockIdx.y * TITAN_TILE + threadIdx.y;
    uint64_t col = blockIdx.x * TITAN_TILE + threadIdx.x;

    float acc = 0.0f;
    uint64_t num_tiles = (k + TITAN_TILE - 1) / TITAN_TILE;

    for (uint64_t t = 0; t < num_tiles; ++t) {
        uint64_t a_col = t * TITAN_TILE + threadIdx.x;
        uint64_t b_row = t * TITAN_TILE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TITAN_TILE; ++i) {
            acc += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}

inline void titan_matmul(float* out, const float* a, const float* b,
                          uint64_t m, uint64_t n, uint64_t k,
                          cudaStream_t stream = 0) {
#ifdef TITAN_USE_CUBLAS
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major; compute out^T = b^T * a^T to get row-major
    // out = a * b without explicit transposition.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)n, (int)m, (int)k,
                &alpha, b, (int)n, a, (int)k,
                &beta, out, (int)n);
#else
    dim3 block(TITAN_TILE, TITAN_TILE);
    dim3 grid((unsigned)((n + TITAN_TILE - 1) / TITAN_TILE),
              (unsigned)((m + TITAN_TILE - 1) / TITAN_TILE));
    titan_matmul_kernel<<<grid, block, 0, stream>>>(out, a, b, m, n, k);
#endif
}

// ---------------------------------------------------------------------
// Attention: naive scaled dot-product, one block per (head, query position)
// ---------------------------------------------------------------------
__global__ void titan_attention_kernel(float* __restrict__ out,
                                        const float* __restrict__ q,
                                        const float* __restrict__ k,
                                        const float* __restrict__ v,
                                        uint64_t heads, uint64_t head_dim, uint64_t seq_len) {
    extern __shared__ float scores[]; // seq_len floats

    uint64_t h = blockIdx.y;
    uint64_t i = blockIdx.x; // query position
    uint64_t head_off = h * seq_len * head_dim;
    float scale = rsqrtf((float)head_dim);

    // Compute scores[j] = scale * dot(q_i, k_j)
    for (uint64_t j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float dot = 0.0f;
        for (uint64_t d = 0; d < head_dim; ++d) {
            dot += q[head_off + i * head_dim + d] * k[head_off + j * head_dim + d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Softmax over scores (single thread for simplicity; seq_len assumed small/medium)
    if (threadIdx.x == 0) {
        float max_val = -INFINITY;
        for (uint64_t j = 0; j < seq_len; ++j) max_val = fmaxf(max_val, scores[j]);
        float sum = 0.0f;
        for (uint64_t j = 0; j < seq_len; ++j) {
            scores[j] = __expf(scores[j] - max_val);
            sum += scores[j];
        }
        for (uint64_t j = 0; j < seq_len; ++j) scores[j] /= sum;
    }
    __syncthreads();

    // out_i = sum_j scores[j] * v_j
    for (uint64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint64_t j = 0; j < seq_len; ++j) {
            acc += scores[j] * v[head_off + j * head_dim + d];
        }
        out[head_off + i * head_dim + d] = acc;
    }
}

inline void titan_attention(float* out, const float* q, const float* k, const float* v,
                             uint64_t heads, uint64_t head_dim, uint64_t seq_len,
                             cudaStream_t stream = 0) {
    dim3 grid((unsigned)seq_len, (unsigned)heads);
    int threads = 128;
    size_t shmem = seq_len * sizeof(float);
    titan_attention_kernel<<<grid, threads, shmem, stream>>>(out, q, k, v, heads, head_dim, seq_len);
}

// ---------------------------------------------------------------------
// Elementwise ops
// ---------------------------------------------------------------------
__global__ void titan_elementwise_kernel(float* __restrict__ out,
                                          const float* __restrict__ in,
                                          uint64_t len, int op) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= len) return;
    float x = in[i];
    switch (op) {
        case 0: out[i] = x; break;                              // id
        case 1: out[i] = x > 0.0f ? x : 0.0f; break;             // relu
        case 2: {                                                 // gelu
            const float c = 0.7978845608028654f;
            float inner = c * (x + 0.044715f * x * x * x);
            out[i] = 0.5f * x * (1.0f + tanhf(inner));
            break;
        }
        case 3: out[i] = x / (1.0f + __expf(-x)); break;         // silu
        default: out[i] = x; break;
    }
}

enum TitanElementwiseOp { TITAN_OP_ID = 0, TITAN_OP_RELU = 1, TITAN_OP_GELU = 2, TITAN_OP_SILU = 3 };

inline void titan_elementwise(float* out, const float* in, uint64_t len, int op,
                               cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (int)((len + threads - 1) / threads);
    titan_elementwise_kernel<<<blocks, threads, 0, stream>>>(out, in, len, op);
}

inline void titan_barrier(cudaStream_t stream = 0) {
    cudaStreamSynchronize(stream);
}
