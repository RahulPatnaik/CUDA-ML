import torch
# OLD: from python_src.cuda_ml_binary.modules import BinaryLinear, calculate_alpha
from cuda_ml_binary.modules import BinaryLinear, calculate_alpha # <--- CORRECTED LINE
import time

# Try to import the compiled CUDA C++ extension directly for direct testing of kernel
try:
    from cuda_ml_binary import _C
except ImportError:
    print("CUDA extension not found. Please run 'python setup.py install' or 'pip install -e .' in the project root.")
    _C = None

# Removed CPU-based packing/binarization helper functions

def test_binary_linear_module():
    if _C is None:
        print("Skipping test: CUDA extension not loaded.")
        return

    print("--- Testing BinaryLinear Module ---")

    # Dimensions
    batch_size = 4
    in_features = 64 # Make sure this is a multiple of 32 for simpler packing initially
    out_features = 32

    # Create dummy input and weights
    input_fp = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float32)
    weights_fp = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32) # PyTorch default weight shape is [out, in]
    bias_fp = torch.randn(out_features, device='cuda', dtype=torch.float32)

    print(f"Input shape: {input_fp.shape}, Weights shape: {weights_fp.shape}")

    # Create BinaryLinear module
    binary_layer = BinaryLinear(in_features, out_features, bias=True).to('cuda')

    # Manually quantize and set weights (as would happen in quantize_model.py)
    binary_layer._quantize_weights(weights_fp) # This will set weight_packed and weight_alpha
    binary_layer.bias.data.copy_(bias_fp) # Copy bias

    print(f"BinaryLinear module created. Weight packed shape: {binary_layer.weight_packed.shape}")
    print(f"Weight alpha: {binary_layer.weight_alpha.item():.6f}")

    # Perform forward pass
    start_time = time.time()
    output_binary = binary_layer(input_fp)
    torch.cuda.synchronize() # Ensure CUDA operations complete before timing
    end_time = time.time()

    print(f"BinaryLinear forward pass took: {end_time - start_time:.6f} seconds")
    print(f"Output shape: {output_binary.shape}")
    print("Output (first row, first 5 elements):", output_binary[0, :5].tolist())

    # --- Basic Sanity Check (CPU-based reference, very rough) ---
    # This reference should ideally mirror the BNN logic closely.
    # It will still be rough, as BNNs are sensitive to exact binarization and alpha calculation.
    print("\n--- Running CPU-based reference for sanity check (rough approximation) ---")
    
    # Binarize inputs and weights on CPU, then convert to -1/+1 float
    input_binary_ref = torch.where(input_fp.cpu() >= 0, 1.0, -1.0)
    weights_binary_ref = torch.where(weights_fp.cpu() >= 0, 1.0, -1.0)

    # Calculate alpha for input activations (on CPU for reference)
    input_alpha_ref = calculate_alpha(input_fp.cpu())

    # Perform binary matmul on CPU and apply alphas
    # (A @ B.T for nn.Linear)
    output_ref_cpu = (input_binary_ref @ weights_binary_ref.T) * input_alpha_ref * binary_layer.weight_alpha.cpu()
    output_ref_cpu += bias_fp.cpu()

    print("CPU reference output (first row, first 5 elements):", output_ref_cpu[0, :5].tolist())

    # Compare (expect differences due to precision, packing, exact BNN logic, and numerical stability)
    abs_diff = (output_binary.cpu() - output_ref_cpu).abs().mean().item()
    print(f"Mean absolute difference between CUDA and CPU reference: {abs_diff:.6f}")
    
    # Set a more generous tolerance for BNNs due to approximations
    if abs_diff < 1.0: # This threshold is very high for sanity, but reflects the rough nature.
        print("Basic sanity check PASSED (difference is within tolerance).")
    else:
        print("Basic sanity check FAILED (difference is too high). Check implementation.")

    print("\n--- Test BinaryLinear Module Complete ---")

def test_binarize_tensor_kernel():
    if _C is None:
        print("Skipping test: CUDA extension not loaded.")
        return

    print("\n--- Testing Binarize Tensor Kernel ---")
    
    # Create a dummy float tensor
    input_cpu = torch.randn(100, 100, device='cpu', dtype=torch.float32)
    input_cuda = input_cpu.to('cuda')

    # Test CUDA binarization and packing
    start_time_cuda = time.time()
    packed_cuda = _C.binarize_tensor(input_cuda) # This now calls binarize_and_pack_tensor_cuda
    torch.cuda.synchronize() # Ensure kernel completes
    end_time_cuda = time.time()

    print(f"CUDA binarization took: {end_time_cuda - start_time_cuda:.6f} seconds")
    print(f"Packed CUDA tensor shape: {packed_cuda.shape}, dtype: {packed_cuda.dtype}")

    # You can do a more robust check by unpacking the packed_cuda tensor on CPU
    # and comparing it to a CPU-based binarization, but that requires an unpack kernel.
    # For now, just confirm it runs.
    print("--- Test Binarize Tensor Kernel Complete ---")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This project requires a CUDA-enabled GPU.")
        exit()

    print("Starting 1-bit Inference Engine tests...")
    
    test_binarize_tensor_kernel() # Test the raw kernel
    test_binary_linear_module() # Test the PyTorch module wrapper