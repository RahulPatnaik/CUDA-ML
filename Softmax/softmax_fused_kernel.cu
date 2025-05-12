#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256

// NOTE: input_ptr and output_ptr here come from torch::Tensor.data_ptr<float>()
__global__ void softmax_fused_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t N) {
    extern __shared__ float shared[];
    float* shared_sum = shared;       // size = 1
    float* shared_max = shared + 1;   // size = 1

    // Phase 1: compute max
    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x)
        local_max = fmaxf(local_max, input[i]);

    // warp‐reduce max
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));

    if ((threadIdx.x & (warpSize-1)) == 0) {
        atomicMax((int*)shared_max, __float_as_int(local_max));
    }
    __syncthreads();

    float max_val = __int_as_float(*((int*)shared_max));

    // Phase 2: compute exp(x - max) & partial sum
    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        float v = expf(input[i] - max_val);
        output[i] = v; 
        local_sum += v;
    }

    // warp‐reduce sum
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    if ((threadIdx.x & (warpSize-1)) == 0) {
        atomicAdd(shared_sum, local_sum);
    }
    __syncthreads();

    float sum_val = *shared_sum;

    // Phase 3: normalize
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x)
        output[i] /= sum_val;
}

void softmax_fused_cuda(torch::Tensor input, torch::Tensor output) {
    const auto N = input.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = 1;
    // 2 floats in shared (max + sum)
    const int shared_mem = 2 * sizeof(float);

    softmax_fused_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_fused", &softmax_fused_cuda, "Fused Softmax (CUDA)");
}
