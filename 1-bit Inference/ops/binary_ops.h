#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint> // For uint32_t

// Macro for CUDA error checking (add this at the top of your .cu file too)
#define CUDA_CHECK_LAST_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
}

// Forward declarations for CUDA kernels

// Kernel for binarizing float tensor to 1-bit packed uint32 tensor ({-1, +1} mapping)
// Input: float* input_ptr (full-precision input tensor)
// Output: uint32_t* output_packed_ptr (each uint32_t holds 32 1-bit values)
void binarize_and_pack_tensor_cuda(
    torch::Tensor input, // Input tensor (float or half)
    torch::Tensor output_packed // Output tensor (uint32_t)
);

// Kernel for 1-bit Matrix Multiplication (Binarized GEMM)
// C = alpha_A * alpha_B * (A_binary @ B_binary)
// A_packed: [M, K_packed_A] where K_packed_A = ceil(K_A/32)
// B_packed: [N, K_packed_B] where K_packed_B = ceil(K_B/32) -- typically N=out_features, K_B=in_features
// We will compute A @ B.T implicitly where B is [N, K]
void binary_matmul_cuda(
    torch::Tensor A_packed,    // Input activations packed [M, K_packed_A]
    torch::Tensor B_packed,    // Weights packed [N, K_packed_B]
    torch::Tensor output,      // Output [M, N] (float)
    float alpha_A,             // Scaling factor for input activations
    float alpha_B,             // Scaling factor for weights
    int M, int N, int K        // Original float dimensions (M, N, K)
);

// Python bindings for C++ functions
at::Tensor binarize_and_pack_tensor_py(at::Tensor input);
at::Tensor binary_matmul_py(at::Tensor A_packed, at::Tensor B_packed, float alpha_A, float alpha_B, int M, int N, int K);