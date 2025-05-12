from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_softmax_ext",
    ext_modules=[
        CUDAExtension(
            name="cuda_softmax_ext",
            sources=[
                "softmax_fused.cpp",
                "softmax_fused_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
