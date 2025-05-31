CUDA-ML
=======

A learning repository implementing machine learning modules with CUDA acceleration. This project focuses on efficient GPU-based implementations of common ML algorithms.

Contents
--------

-   Optimized CUDA implementations of ML algorithms

-   Performance benchmarks against CPU versions

-   Examples and usage documentation

Current Modules
---------------

-   **Softmax**: GPU-accelerated softmax regression implementation

-   **Convolution**: GPU-accelerated 2D Convolution kernels, outperforming PyTorch on matrices smaller than 256x256 on single GPU inferencing.

-   **1-bit Inference**: Currently loads the DistillBERT model from the huggingface transformers library for 1-bit quantization during inference. (Work in progress)

-   (Additional modules will be added as development progresses)

Requirements
------------

-   NVIDIA GPU with CUDA support

-   CUDA Toolkit (version 12.6)

-   Python 3.10+

-   PyTorch (for reference implementations)

Usage
-----

Each module contains:

-   CUDA kernel source code (.cu files)

-   Python bindings

-   Example scripts

-   Performance tests