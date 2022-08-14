from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 头文件目录
include_dirs = "./src"
# 源代码目录
source = ["./src/MySigmoid.cpp"]

setup(
    name='mysigmoid',  # 模块名称，宏TORCH_EXTENSION_NAME的值
    version="0.1",
    ext_modules=[CppExtension('mysigmoid', sources=source, include_dirs=[include_dirs]),],
    cmdclass={'build_ext': BuildExtension}
)