// main.cu - host harness for generated/generated_megakernel.cu
//
// Allocates device buffers for every tensor in TensorSet (uniform
// TITAN_TENSOR_LEN elements, matching the demo schedule), runs
// run_megakernel on one stream, and prints the result.

#include "generated_megakernel.cu"
#include <cstdio>
#include <vector>

int main() {
    const uint64_t len = TITAN_TENSOR_LEN;
    const size_t bytes = len * sizeof(float);

    TensorSet t{};

    // Allocate every field of TensorSet. Field order must match the
    // generated struct; we allocate via a small helper to avoid
    // hand-listing names twice -- but since C++ has no reflection,
    // generated_megakernel.cu also emits TITAN_TENSOR_NAMES for this
    // harness to iterate.
    float** fields[] = {
#include "tensor_fields.inc"
    };
    size_t n_fields = sizeof(fields) / sizeof(fields[0]);

    for (size_t i = 0; i < n_fields; ++i) {
        TITAN_CUDA_CHECK(cudaMalloc(fields[i], bytes));
        TITAN_CUDA_CHECK(cudaMemset(*fields[i], 0, bytes));
    }

    // Seed the first input tensor (alphabetically first field, by
    // convention the schedule's source node) with 1..len.
    std::vector<float> h_init(len);
    for (uint64_t i = 0; i < len; ++i) h_init[i] = (float)(i + 1);
    TITAN_CUDA_CHECK(cudaMemcpy(*fields[0], h_init.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    TITAN_CUDA_CHECK(cudaStreamCreate(&stream));

    run_megakernel(t, stream);

    TITAN_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Print every tensor.
    std::vector<float> h_out(len);
    for (size_t i = 0; i < n_fields; ++i) {
        TITAN_CUDA_CHECK(cudaMemcpy(h_out.data(), *fields[i], bytes, cudaMemcpyDeviceToHost));
        printf("tensor[%zu] = ", i);
        for (uint64_t j = 0; j < len; ++j) printf("%.4f ", h_out[j]);
        printf("\n");
    }

    for (size_t i = 0; i < n_fields; ++i) cudaFree(*fields[i]);
    cudaStreamDestroy(stream);
    return 0;
}
