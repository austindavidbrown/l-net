# Build registration
tools::package_native_routine_registration_skeleton("./R-package", "R-package/src/init.c")

# Document
R CMD INSTALL ./R-package # needs to install package to document
devtools::document("./R-package")
# roxygen2::roxygenise("./R-package")

# Generate reference manual
system("R CMD Rd2pdf ./R-package --force -o pros-manual.pdf")

# Check
system("R CMD check --as-cran ./R-package")
devtools::check("./R-package")

devtools::install("R-package")