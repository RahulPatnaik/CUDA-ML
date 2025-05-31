# debug_import.py
import sys
import os

print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    # Try importing the top-level package directly
    import cuda_ml_binary
    print("Successfully imported cuda_ml_binary package.")

    # Try importing the _C extension
    from cuda_ml_binary import _C
    print("Successfully imported _C (CUDA extension).")
    
    # Test a function from _C
    # Ensure torch is imported and CUDA is available before this part
    import torch
    if torch.cuda.is_available():
        test_tensor = torch.randn(32, 32, device='cuda')
        # Call a simple function like binarize_tensor
        packed_tensor = _C.binarize_tensor(test_tensor)
        print(f"Test call to _C.binarize_tensor successful. Output shape: {packed_tensor.shape}")
    else:
        print("CUDA not available for _C test.")

except ImportError as e:
    print(f"ImportError: {e}")
    print("This indicates a problem with how the package is installed or its path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\n--- Debug Import Test Complete ---")