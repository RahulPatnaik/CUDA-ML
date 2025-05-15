from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="conv2d_cuda",
    ext_modules=[
        CUDAExtension(
            name="conv2d_cuda",
            sources=["conv2d_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3","-std=c++14"],
                "nvcc": ["-O3","--use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
