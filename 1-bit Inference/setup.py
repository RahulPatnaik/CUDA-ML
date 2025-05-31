from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_ml_binary',
    version='0.1.0',
    packages=find_packages(where='python_src'),
    package_dir={'': 'python_src'},
    ext_modules=[
        CUDAExtension(
            name='cuda_ml_binary._C', # This name will be imported as `from cuda_ml_binary import _C`
            sources=[
                'ops/binary_ops.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    description='A 1-Bit GPU Inference Engine for Transformers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Rahul Patnaik', # Replace with your name
    author_email='rpatnaik2005@gmail.com', # Replace with your email
    url='https://github.com/RahulPatnaik/CUDA-ML-Binary', # Update with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or whatever license you choose
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Hardware',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.10.0',
        'transformers>=4.0.0',
        'numpy>=1.20.0',
    ],
    extra_compile_args={
    'cxx': ['-O3'],
    'nvcc': ['-O3', '--use_fast_math', '-g'] # Add -g here
    }
)