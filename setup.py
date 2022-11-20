from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="sample_neighber",
    ext_modules=[
        CppExtension("sample_neighber", sources=["spikenet/sample_neighber.cpp"], extra_compile_args=['-g']),

    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
