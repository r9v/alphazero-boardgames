from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "mcts.c_mcts",
        ["mcts/c_mcts.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "games.santorini.c_game",
        ["games/santorini/c_game.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="alphazero-cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
