from setuptools import setup, Extension
import pybind11

ext = Extension(
    "two_opt_cpp",
    ["two_opt_cpp.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"]
)

setup(
    name="two_opt_cpp",
    version="0.1",
    ext_modules=[ext]
)

