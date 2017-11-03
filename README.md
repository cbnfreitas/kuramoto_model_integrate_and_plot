# kuramoto_model_integrate_and_plot

Check run_integrate_measure_and_plot.ipynb!

This repository provides Python code to perform numerical integration of the Kuramoto Model, a network of coupled non-identical phase oscillators. It allows simulations via odeint from scipy.integrate (ODEs) or ddeint (DDEs), in the presence of overall communication delay between oscillators.

We also configure and present suggestions for figures which are suitable for Latex (see tex_example/Example.pdf).

The Kuramoto vector field is defined in network_of_oscillators_Kuramoto_model.py and its parameters are loaded from the /parameters folder.

We require several packages (numpy, scipy, matplotlib, etc). The standard installation of https://anaconda.org/anaconda/python contains all of them.

TODO: unit tests, complete documentation.


