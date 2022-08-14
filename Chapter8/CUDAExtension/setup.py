from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mysigmoid2',
    ext_modules=[
        CUDAExtension('mysigmoid2', [
            './src/MySigmoidKernel.cu',
            './src/MySigmoidCUDA.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })