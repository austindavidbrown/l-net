from distutils.core import setup, Extension
import numpy as np

setup(
    name = "lnet",
    version="0.1",
    author = "Austin David Brown",
    author_email = "brow5079@umn.edu",
    url = "github.com/austindavidbrown/l-net",
    license = "GPL3",
    ext_modules = [
      Extension(
      "lnet",
      sources=["lnet_interface.cpp"],
      language="C++", 
      include_dirs = [np.get_include(), "./include/eigen"], 
      extra_compile_args = ["-Wall", "-std=c++17", "-O3", "-march=native"])
    ])