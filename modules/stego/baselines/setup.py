from setuptools import setup
from Cython.Build import cythonize
import os

setup(
    ext_modules=cythonize(os.path.join("baselines","discop.pyx"),
                            annotate=False,
                            compiler_directives={
                                'boundscheck': False,
                                'language_level': 3
                            })
)