import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the compiled CUDA C++ extension
try:
    from cuda_ml_binary import _C
except ImportError:
    print("CUDA extension not found. Please run 'python setup.py install' or 'pip install -e .' in the project root.")
    _C = None # Fallback for development if not compiled yet

# --- Helper Functions (Python side for calculating alpha) ---

def calculate_alpha(x: torch.Tensor):
    """
    Calculates the scaling factor (alpha) for a tensor.
    Alpha = mean(abs(x)). This works on CUDA tensors.
    """
    # Ensure numerical stability for empty tensors or tensors with all zeros
    if x.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return x.abs().mean()


# --- PyTorch Modules for 1-bit Inference ---

class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weights will be stored as packed 1-bit values and a float alpha
        # Weight shape for matmul is [out_features, ceil(in_features/32)]
        self.register_buffer('weight_packed', None) # uint32_t tensor
        self.register_buffer('weight_alpha', None)  # float scalar

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias) # Initialize bias to zeros for simplicity

    def _quantize_weights(self, weights_fp: torch.Tensor):
        """
        Binarizes and packs the full-precision weights using CUDA.
        This is called during model quantization/conversion.
        `weights_fp` is expected to be [out_features, in_features]
        """
        if _C is None:
            raise RuntimeError("CUDA extension not loaded. Cannot quantize weights.")

        # Ensure weights are on GPU and contiguous
        weights_fp_cuda = weights_fp.to(weights_fp.device).contiguous()

        # Calculate alpha for weights
        weight_alpha = calculate_alpha(weights_fp_cuda)

        # Binarize and pack weights using CUDA kernel
        # _C.binarize_tensor expects flat input for packing, then reshape for matmul
        # It takes [out_features, in_features] -> flat and packs
        packed_flat = _C.binarize_tensor(weights_fp_cuda)

        # Reshape to [out_features, (in_features + 31) // 32]
        packed_reshaped = packed_flat.reshape(
            self.out_features, (self.in_features + 31) // 32
        )

        # Register as buffer
        self.register_buffer('weight_packed', packed_reshaped)
        self.register_buffer('weight_alpha', weight_alpha)

        # Clear original parameter if it was one
        if hasattr(self, 'weight'):
            del self.weight
            self.register_buffer('weight', None)


    def forward(self, input: torch.Tensor):
        if _C is None:
            raise RuntimeError("CUDA extension not loaded. Cannot perform binary operations.")

        # Ensure input is on CUDA and contiguous
        input_cuda = input.to(input.device).contiguous()

        # --- Handle potential 3D input (Batch, Sequence Length, Features) ---
        original_shape = input_cuda.shape
        M_total = 1 # Total rows for the binary_matmul operation
        
        if len(original_shape) == 3:
            # For Transformer layers, input is typically [batch_size, sequence_length, hidden_size]
            # Flatten to [batch_size * sequence_length, hidden_size] for linear layer
            # M_total will be batch_size * sequence_length
            input_flat_for_matmul = input_cuda.view(-1, self.in_features)
            M_total = input_flat_for_matmul.shape[0]
        elif len(original_shape) == 2:
            # Input is already [batch_size, features]
            input_flat_for_matmul = input_cuda
            M_total = input_flat_for_matmul.shape[0]
        else:
            raise ValueError(f"Unsupported input shape for BinaryLinear: {original_shape}")


        # Calculate alpha for input activations (using the flattened input)
        input_alpha = calculate_alpha(input_flat_for_matmul)

        # Binarize and pack input activations using CUDA kernel
        # _C.binarize_tensor expects a flat tensor for packing
        input_packed_flat_cuda = _C.binarize_tensor(input_flat_for_matmul)

        # Reshape to [M_total, (in_features + 31) // 32] for binary_matmul
        input_packed_reshaped_cuda = input_packed_flat_cuda.reshape(
            M_total, (self.in_features + 31) // 32
        )

        # Perform 1-bit matrix multiplication using CUDA kernel
        # _C.binary_matmul(A_packed, B_packed, alpha_A, alpha_B, M, N, K_orig)
        # A_packed: input_packed_reshaped_cuda [M_total, K_packed]
        # B_packed: self.weight_packed [out_features, K_packed]
        output_flat = _C.binary_matmul(
            input_packed_reshaped_cuda,
            self.weight_packed,
            input_alpha.item(),
            self.weight_alpha.item(),
            M_total,             # M (Batch * Sequence length for 3D inputs)
            self.out_features,   # N (Output features)
            self.in_features     # K (Input features)
        )

        # --- Reshape output back to original 3D shape if input was 3D ---
        if len(original_shape) == 3:
            # Reshape output_flat (M_total, out_features) back to (Batch, Sequence Length, out_features)
            output_final = output_flat.view(original_shape[0], original_shape[1], self.out_features)
        else: # 2D input
            output_final = output_flat


        if self.bias is not None:
            output_final = output_final + self.bias
        return output_final # Return the reshaped output

class BinaryActivation(nn.Module):
    """
    Binarizes activations to {-1, +1} using sign function.
    For true BNNs, this output would also be packed 1-bit.
    For this project, it's illustrative and currently outputs float{-1, 1}.
    """
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x: torch.Tensor):
        # Apply the BNN sign activation: >=0 -> +1, <0 -> -1
        return torch.where(x >= 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.tensor(-1.0, device=x.device, dtype=x.dtype))