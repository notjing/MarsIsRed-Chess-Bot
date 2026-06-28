from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mcts_exts", # output name
        ["mcts_exts.cpp", "feature_extraction.cpp", "zobristHashing.cpp"],  # input name
    ),
]

setup(
    name="mcts_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
