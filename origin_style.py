import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, NullLocator
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 1.1, 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'axes.axisbelow': True, 'axes.grid': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.major.size': 4.5, 'ytick.major.size': 4.5, 'xtick.minor.size': 2.6, 'ytick.minor.size': 2.6, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7, 'xtick.color': 'black', 'ytick.color': 'black', 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.frameon': True, 'legend.edgecolor': 'black', 'legend.fancybox': False, 'legend.framealpha': 1.0, 'legend.borderpad': 0.4, 'legend.handlelength': 1.6, 'savefig.dpi': 300, 'figure.dpi': 300})

def box(ax):
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.1)
        ax.spines[s].set_color('black')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return ax

def bar(ax, cat_axis='x'):
    box(ax)
    if cat_axis == 'x':
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x', which='minor', top=False, bottom=False)
    else:
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='y', which='minor', left=False, right=False)
    return ax