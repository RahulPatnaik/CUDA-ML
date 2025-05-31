#include "binary_ops.h" // Include the updated header
#include <stdio.h> // For debugging, can remove later

// --- Helper Functions (Device-side) ---

// Map float to 1-bit representation (0 for +1, 1 for -1)
__device__ inline uint32_t get_bit_representation(float val) {
    return (val < 0) ? 1u : 0u; // Map negative to 1, non-negative to 0
}

// Get 1-bit value from packed uint32_t (from bit representation)
// Maps 0 -> +1.0f, 1 -> -1.0f
__device__ inline float get_val_from_bit(uint32_t packed_val, int bit_idx) {
    return ((packed_val >> bit_idx) & 1u) ? -1.0f : 1.0f;
}

// --- CUDA Kernels ---

// Kernel to binarize float tensor and pack it into a 1-bit uint32 tensor
// input_ptr: pointer to float tensor (e.g., activations or weights)
// output_packed_ptr: pointer to uint32_t packed tensor
// N: total number of elements in input tensor
__global__ void binarize_and_pack_tensor_kernel(
    const float* __restrict__ input_ptr,
    uint32_t* __restrict__ output_packed_ptr,
    int N
) {
    // Each thread processes one float element and packs its 1-bit representation
    // into the correct bit position within a uint32_t word.
    // Multiple threads will need to write to the same uint32_t word (different bits).
    // Initialize output_packed_ptr to zeros on host before launching.
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < N) {
        uint32_t bit_val = get_bit_representation(input_ptr[global_idx]);

        int packed_word_idx = global_idx / 32;
        int bit_in_word = global_idx % 32;

        if (bit_val == 1u) { // If the value maps to bit 1 (i.e., -1.0f)
            atomicOr(&output_packed_ptr[packed_word_idx], (1u << bit_in_word));
        }
        // If bit_val is 0, no action needed as the word is initialized to zero
    }
}


// Kernel for 1-bit Matrix Multiplication (Binarized GEMM)
// C = alpha_A * alpha_B * (A_binary @ B_binary.T)
// A_packed_ptr: [M, K_packed_A]
// B_packed_ptr: [N_out, K_packed_in] (PyTorch standard weight layout)
// C_ptr: [M, N]
// M, N, K_orig: original dimensions M (batch), N (out_features), K_orig (in_features)
__global__ void binary_matmul_kernel(const uint32_t* __restrict__ A_packed_ptr,
                                     const uint32_t* __restrict__ B_packed_ptr,
                                     float* __restrict__ C_ptr,
                                     float alpha_A, float alpha_B,
                                     int M, int N, int K_orig) {
    // Determine row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Output row (batch index)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Output col (output feature index)

    if (row < M && col < N) {
        float acc = 0.0f;
        int K_packed = (K_orig + 31) / 32; // Number of uint32_t words per row/column

        // Loop over the 'K' dimension (input features)
        for (int k_word_idx = 0; k_word_idx < K_packed; ++k_word_idx) {
            uint32_t a_val_packed = A_packed_ptr[row * K_packed + k_word_idx];
            // Access B_packed (weights) which is [N, K_packed] - so B_packed[col_idx][k_word_idx]
            uint32_t b_val_packed = B_packed_ptr[col * K_packed + k_word_idx];

            // Perform XNOR-Popcount for 32 bits within the word
            uint32_t xor_result = a_val_packed ^ b_val_packed; // XOR A, B
            int bits_set = __popc(xor_result); // Count set bits (i.e., different signs)

            // Contribution to sum: (number of same signs) - (number of different signs)
            // number of same signs = 32 - bits_set
            // acc += (32 - bits_set) - bits_set = 32 - 2 * bits_set
            acc += (float)(32 - (2 * bits_set));
        }
        C_ptr[row * N + col] = acc * alpha_A * alpha_B;
    }
}


// --- Python Bindings (C++ wrapper functions calling CUDA kernels) ---

at::Tensor binarize_and_pack_tensor_py(at::Tensor input) {
    AT_ASSERTM(input.is_cuda(), "Input must be a CUDA tensor");
    AT_ASSERTM(input.is_contiguous(), "Input must be contiguous");
    AT_ASSERTM(input.dtype() == torch::kFloat || input.dtype() == torch::kHalf, "Input must be float or half");

    int N = input.numel();
    int N_packed = (N + 31) / 32;
    // Output tensor MUST be zero-initialized for atomicOr to work correctly
    at::Tensor output_packed = torch::zeros({N_packed}, input.options().dtype(torch::kUInt32));

    const int BLOCK_SIZE = 256;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    binarize_and_pack_tensor_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), // Assuming float for now
        output_packed.data_ptr<uint32_t>(),
        N
    );
    CUDA_CHECK_LAST_ERROR(); // Check for errors after kernel launch

    return output_packed;
}


at::Tensor binary_matmul_py(at::Tensor A_packed, at::Tensor B_packed, float alpha_A, float alpha_B, int M, int N, int K_orig) {
    AT_ASSERTM(A_packed.is_cuda(), "A_packed must be a CUDA tensor");
    AT_ASSERTM(B_packed.is_cuda(), "B_packed must be a CUDA tensor");
    AT_ASSERTM(A_packed.is_contiguous(), "A_packed must be contiguous");
    AT_ASSERTM(B_packed.is_contiguous(), "B_packed must be contiguous");
    AT_ASSERTM(A_packed.dtype() == torch::kUInt32, "A_packed must be uint32");
    AT_ASSERTM(B_packed.dtype() == torch::kUInt32, "B_packed must be uint32");

    at::Tensor output = torch::zeros({M, N}, A_packed.options().dtype(torch::kFloat)); // Output is float

    // Determine grid and block dimensions for 2D matrix multiplication
    dim3 block_dim(32, 32); // Example block size
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

    binary_matmul_kernel<<<grid_dim, block_dim>>>(
        A_packed.data_ptr<uint32_t>(),
        B_packed.data_ptr<uint32_t>(),
        output.data_ptr<float>(),
        alpha_A,
        alpha_B,
        M, N, K_orig
    );
    CUDA_CHECK_LAST_ERROR(); // Check for errors after kernel launch

    return output;
}

// Register the C++ functions to be exposed to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binarize_tensor", &binarize_and_pack_tensor_py, "Binarize and pack a float/half tensor to 1-bit packed uint32 (CUDA)");
    m.def("binary_matmul", &binary_matmul_py, "1-bit Matrix Multiplication (CUDA)");
}