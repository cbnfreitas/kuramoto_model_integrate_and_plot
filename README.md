# kuramoto_model_integrate_and_plot

This code requires several Python 3 packages (numpy, scipy, matplotlib, etc). We suggest to install them via https://anaconda.org/anaconda/python. 

This repository provides Python code to perform numerical integration of the Kuramoto Model, which is a network of coupled non-identical phase oscillators. It also allows simulations via odeint from scipy.integrate (ODEs) or ddeint (DDEs), in the presence of communication delay between oscillators.

We also configure and present suggestions for figures which are suitable for Latex (see tex_example/Example.pdf).

The Kuramoto vector field is defined in network_of_oscillators_Kuramoto_model.py and its parameters are loaded from the /parameters folder.

The examples can be found at run_integrate_measure_and_plot.ipynb.

TODO: unit tests, complete documentation (shame on me!) and check convergency order of odeint and ddeint.


