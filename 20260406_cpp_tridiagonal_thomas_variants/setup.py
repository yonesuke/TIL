from setuptools import setup, Extension
import pybind11

ext = Extension(
    "tridiagonal_cpp",
    sources=["tridiagonal_cpp.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-std=c++17", "-march=native"],
    language="c++",
)

setup(name="tridiagonal_cpp", ext_modules=[ext])
