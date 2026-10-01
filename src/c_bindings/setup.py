from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11

ext_modules = [
    Pybind11Extension(
        "mcts_exts",
        ["mcts_exts.cpp", "feature_extraction.cpp", "zobristHashing.cpp", "sequentialHalving.cpp"],
        include_dirs=[
            ".",
            "headerFiles",
            pybind11.get_include(),
        ],
        cxx_std=17,
    ),
]

setup(
    name="mcts_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
