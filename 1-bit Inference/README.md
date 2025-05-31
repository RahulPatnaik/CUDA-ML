# CUDA-ML-Binary: A 1-Bit GPU Inference Engine for Transformers

## Project Goal
To build a high-performance 1-bit inference engine in CUDA C++ that can run binarized versions of Hugging Face Transformer models (like DistilBERT or TinyLlama) on NVIDIA GPUs, accessible via a Python module.

## Key Features
- **1-Bit Quantization:** Implements Binary Neural Network (BNN) principles where both weights and activations are quantized to `{-1, +1}`.
- **CUDA C++ Kernels:** Optimized kernels for 1-bit matrix multiplication (GEMM) and activation functions, leveraging GPU parallelism.
- **Python Integration:** Seamlessly integrates with PyTorch via `torch.cuda.cpp_extension` to allow easy binarization and inference of Hugging Face models.
- **Performance Benchmarking:** Comprehensive evaluation of memory savings, inference latency, and throughput compared to full-precision models. (To be implemented)
- **Accuracy Evaluation:** Analysis of accuracy trade-offs on standard NLP benchmarks (e.g., GLUE for DistilBERT). (To be implemented)

## Why 1-Bit?
1-bit quantization offers unparalleled memory reduction (up to 32x compared to FP32) and significant potential for inference speedups, enabling the deployment of large language models on resource-constrained devices like personal GPUs or edge hardware.

## Project Structure

```
.

├── ops/ # CUDA C++ kernels and headers

│ ├── binary_ops.h # CUDA kernel declarations

│ └── binary_ops.cu # CUDA kernel implementations and Python bindings

├── python_src/ # Python modules for integration

│ └── cuda_ml_binary/ # The core Python package

│ ├── init.py # Package initialization

│ ├── modules.py # PyTorch nn.Module wrappers (BinaryLinear, BinaryActivation)

│ └── quantize_model.py # Script to convert HF models to 1-bit

├── examples/ # Usage examples and testing scripts

│ ├── test_binary_linear.py # Basic test for BinaryLinear module and CUDA kernels

│ └── run_quantized_bert.py # (Future: dedicated script for running specific quantized BERTs)

├── setup.py # Build script for the CUDA extension

├── requirements.txt # Python dependencies

└── .gitignore # Git ignore rules
```

## Setup & Installation (Reproducible Steps)

Follow these steps precisely to set up your environment and install the necessary dependencies, including building the custom CUDA extension. This guide assumes you have `conda` installed.

1.  **Navigate to your Project Directory:**
    Open your terminal and change into your project's root directory:
    ```bash
    cd ~/Desktop/CUDA ML/1-bit Inference main/
    ```

2.  **Create a Conda Environment with Python 3.11:**
    It's crucial to use a Python version compatible with PyTorch's CUDA binaries (e.g., 3.11).
    ```bash
    conda create -n binary_ml_env python=3.11
    ```
    Confirm when prompted.

3.  **Activate the Conda Environment:**
    ```bash
    conda activate binary_ml_env
    ```
    Your terminal prompt should change (e.g., `(binary_ml_env)`).

4.  **Install PyTorch with CUDA Support:**
    **This is the most critical step.** You *must* install PyTorch with CUDA support that matches your system's CUDA toolkit version.
    *   **Determine your CUDA Version:** In a new terminal (or after `conda deactivate`), run `nvcc --version` or `nvidia-smi` to find your CUDA version (e.g., CUDA 11.8, 12.1, 12.4, etc.).
    *   **Get PyTorch Install Command:** Visit the official PyTorch website's "Get Started Locally" section: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
        *   Select "Stable", your OS (Linux), "Conda", "Python 3.11", and your detected CUDA version.
        *   Copy the exact `conda install` command provided.
    *   **Execute the Command (Example for CUDA 11.8):**
        ```bash
        # Replace with the exact command from PyTorch website for your CUDA version
        conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
        ```

5.  **Install General Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Install `ninja` (Recommended for Faster Compilation):**
    `ninja` is a faster build system for C++/CUDA extensions.
    ```bash
    conda install ninja
    ```

7.  **Install Conda's C++ Compiler Toolchain (`gxx_linux-64`):**
    This helps prevent runtime `libstdc++` version errors by ensuring a consistent compiler toolchain within your Conda environment.
    ```bash
    conda install -c conda-forge gxx_linux-64
    ```

8.  **Prepare Project Structure for Compilation:**
    Ensure the `cuda_ml_binary` package directory exists within `python_src/` and move your Python modules into it.
    ```bash
    mkdir -p python_src/cuda_ml_binary
    mv python_src/__init__.py python_src/cuda_ml_binary/
    mv python_src/modules.py python_src/cuda_ml_binary/
    mv python_src/quantize_model.py python_src/cuda_ml_binary/
    ```

9.  **Clean Previous Build Artifacts:**
    It's crucial to remove any leftover compiled files from previous failed attempts to ensure a clean rebuild.
    ```bash
    rm -rf build
    rm -rf python_src/cuda_ml_binary.egg-info
    rm -f python_src/cuda_ml_binary/_C*.so
    find . -name "__pycache__" -exec rm -rf {} +
    ```

10. **Build and Install the Custom CUDA Extension:**
    This command compiles your `ops/binary_ops.cu` and `ops/binary_ops.h` into a Python-importable shared library (`.so` file) and links it as an editable package.
    ```bash
    pip install -e .
    ```

    *Expected output:* You should see compilation messages for `nvcc` and `g++`, and the installation should complete without errors.

---

## Usage

Once the installation is complete, you can run the provided example scripts.

1.  **Activate your Conda environment:**
    ```bash
    conda activate binary_ml_env
    ```

2.  **Run the `test_binary_linear.py` script:**
    This script tests the core `BinaryLinear` module and its underlying CUDA kernels with dummy data. It verifies that your 1-bit matrix multiplication and packing operations are functioning correctly.
    ```bash
    python examples/test_binary_linear.py
    ```
    *Expected output:* You should see various debug messages, tensor shapes, and a confirmation that the basic sanity check passed.

3.  **Run the `quantize_model.py` script:**
    This script loads a pre-trained DistilBERT model from Hugging Face, quantizes its linear layers to your 1-bit format, and then runs a minimal inference.
    ```bash
    python python_src/cuda_ml_binary/quantize_model.py
    ```
    *Expected output:* You will see messages about downloading the DistilBERT model, then a list of `Linear` layers being quantized. Finally, you'll see a sample output from the quantized model's inference (the numerical values will likely be random as the model is not trained for 1-bit).

---
