from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='preprocess_extension',
    ext_modules=[
        CppExtension(
            'preprocess_extension',
            ['sgt_preprocess.cpp'],
            extra_compile_args=['-fopenmp'],  # 添加 OpenMP 编译选项
            extra_link_args=['-fopenmp'],     # 添加 OpenMP 链接选项
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)