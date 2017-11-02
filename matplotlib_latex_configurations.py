# Configurations to save Matplotlib figures
import matplotlib.pyplot as plt
from pylab import rcParams

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

golden_ration = (1 + 5 ** 0.5) / 2
one_column_figure_size = 1.7
rcParams['figure.figsize'] = (one_column_figure_size * golden_ration, one_column_figure_size)
rcParams['axes.linewidth'] = 0.25
rcParams['xtick.major.width'] = 0.25
rcParams['ytick.major.width'] = 0.25