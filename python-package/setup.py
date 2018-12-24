from setuptools import setup, Extension
import numpy as np

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "l-net",
    version="0.1.1",
    author = "Austin David Brown",
    author_email = "brow5079@umn.edu",
    url = "https://github.com/austindavidbrown/l-net",
    description="Adds the ability to combine l1 to l10 penalties in regression extending the elastic-net.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "GPL3",
    ext_modules = [
      Extension(
      "lnet",
      sources=["lnet_interface.cpp"],
      language="C++", 
      include_dirs = [np.get_include(), "./include/eigen"], 
      extra_compile_args = ["-Wall", "-std=c++17", "-O3", "-march=native"])
    ])