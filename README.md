Penalized Regression on Steroids
===========

Adds the ability to combine l1 to l10 penalties in regression extending the elastic-net.

Python Installation
-------
```bash
pip install git+https://github.com/austindavidbrown/pros/#egg=pros\&subdirectory=python-package
```

R Installation
-------

Add this to your `~/.R/Makevars` file to improve performance:

```bash
CXXFLAGS = -O3 -march=native -mfpmath=sse
```

and install with

```r
install.packages("devtools") # Install devtools from CRAN
devtools::install_github("austindavidbrown/pros/R-package")
```

Documentation
-------
See help for the Python package.

See the [Reference Manual](https://github.com/austindavidbrown/pros/blob/master/pros-manual.pdf) for the R package.

License
-------
The code is licensed under GPL-3.

References
---------
- This is for a project in STAT8053 at the University of Minnesota.

