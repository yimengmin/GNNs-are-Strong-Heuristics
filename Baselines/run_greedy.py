from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

# pybind11 include path helper
class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        "greedy_tsp",
        ["greedy_tsp.cpp"],
        include_dirs=[get_pybind_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="greedy_tsp",
    version="0.0.1",
    author="",
    description="Greedy TSP (NN) baseline with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

