#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        const cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                           \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";    \
            std::cerr << "code: " << error << ", reason: "                    \
                      << cudaGetErrorString(error) << std::endl;              \
            exit(1);                                                          \
        }                                                                     \
    }

__global__ void find_max_kernel(const float* input, float* max_val, int N) {
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = input[0];
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        atomicMax((int*)&shared_max, __float_as_int(input[i]));
    }
    __syncthreads();

    if (threadIdx.x == 0)
        *max_val = shared_max;
}

__global__ void compute_exp_kernel(const float* input, float* exp_out, float* max_val, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        exp_out[i] = expf(input[i] - *max_val);
    }
}

__global__ void normalize_kernel(float* exp_out, float* output, float* sum_exp, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = exp_out[i] / (*sum_exp);
    }
}

__global__ void sum_exp_kernel(float* exp_out, float* sum_out, int N) {
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        atomicAdd(&shared_sum, exp_out[i]);
    }
    __syncthreads();

    if (threadIdx.x == 0)
        *sum_out = shared_sum;
}

void softmax_stable(const float* input_host, float* output_host, int N) {
    float *input_dev, *exp_dev, *output_dev;
    float *max_dev, *sum_dev;

    CHECK_CUDA(cudaMalloc((void**)&input_dev, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&exp_dev, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&output_dev, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&max_dev, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&sum_dev, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(input_dev, input_host, N * sizeof(float), cudaMemcpyHostToDevice));

    find_max_kernel<<<1, 256>>>(input_dev, max_dev, N);
    compute_exp_kernel<<<1, 256>>>(input_dev, exp_dev, max_dev, N);
    sum_exp_kernel<<<1, 256>>>(exp_dev, sum_dev, N);
    normalize_kernel<<<1, 256>>>(exp_dev, output_dev, sum_dev, N);

    CHECK_CUDA(cudaMemcpy(output_host, output_dev, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(input_dev);
    cudaFree(exp_dev);
    cudaFree(output_dev);
    cudaFree(max_dev);
    cudaFree(sum_dev);
}

int main() {
    const int N = 8;
    float input[N] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f, 2.5f, 0.0f, 1.5f};
    float output[N];

    softmax_stable(input, output, N);

    std::cout << "Softmax output:" << std::endl;
    for (int i = 0; i < N; ++i)
        std::cout << output[i] << " ";
    std::cout << std::endl;

    return 0;
}
