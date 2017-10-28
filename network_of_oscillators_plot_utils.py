import matplotlib.pyplot as plt
import numpy as np

import networkx as nx
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker

from network_of_oscillators_integrate_and_sync_quantifiers import *


def chop_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#This cmap is applied in to node coloring 
cmap_aux = LinearSegmentedColormap.from_list("", ["cyan","#580058"])

#In gray scale, cyan becomes almost white, this is why we chop the begining of the color map
cyan_purple_cmap = chop_colormap(cmap_aux, 0.05, 1)

def frequency_to_color(w):
    colors_cyan_purple = cyan_purple_cmap(np.linspace(0, 1, 1001))
    w_min = min(w)
    w_max = max(w)
    return [colors_cyan_purple[int(1000*(w[i] - w_min)/(w_max - w_min))] for i in range(len(w))]

def plot_coupling_graph(E, w, ax=None):
    if ax is None:
        ax = plt.gca()

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([0, 0], [1,1], linewidth = None)

    G=nx.Graph()
    for i in range(len(E)):
        G.add_edge(int(E[i,0]), int(E[i,1]))

    pos=nx.kamada_kawai_layout(G)

    nx.draw_networkx(G,  
                     pos = pos, 
                     node_color = frequency_to_color(w), with_labels=False,
                     node_size=100,
                     edge_color='gray', ax = ax)
    
    ax.collections[0].set_edgecolor("black") #Drawing a black border around nodes.
    ax.collections[0].set_linewidth(0.5)
    
    ax.collections[1].set_linewidth(0.5)  # Change edges linewidth

    #Creates node labels in white with a black border
    for i in range(max(pos)+1):
        text = ax.text(pos[i][0], pos[i][1], i, color='white', size = 8,
                                  ha='center', va='center')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])    
        
    sm = plt.cm.ScalarMappable(cmap=cyan_purple_cmap, norm=plt.Normalize(vmin=min(w), vmax=max(w)))
    sm._A = []
    cb1 = plt.colorbar(sm)
    cb1.set_label(r'$\omega_i$')
    
    return ax
    
def mod2pi(y):
    maxdif=max(abs(np.diff(y)))
    ymod=np.mod(y, 2*np.pi)
    pos=np.where(abs(np.diff(ymod)) > maxdif)[0]
    ymod[pos]=np.nan
    return ymod

def get_color_from_list(i, color = plt.rcParams['axes.prop_cycle'].by_key()['color']): 
    if not type(color) is list: color = [color]
    return color[np.mod(i, len(color))]


def plot_phase_minus_psi(t, theta, psi, tlim = None,
                         shift_axis = False, 
                         color = plt.rcParams['axes.prop_cycle'].by_key()['color'],
                         sel = None,
                         show_right_ax_labels = True,
                         ax=None):
    if ax is None:
        ax = plt.gca()

    n = np.shape(theta)[1]
    
    if tlim is None:
        t_indexes = range(len(t))
    else:
        t_indexes = np.intersect1d(np.where(t>=tlim[0])[0], np.where(t<=tlim[1])[0])
        
    if sel is None:
        sel_color = range(n)
        sel = range(n)
    else:
        if color == plt.rcParams['axes.prop_cycle'].by_key()['color']:
            sel_color = range(len(sel))
        else:
            sel_color = sel


    shift = 0
    if shift_axis:
        shift = 1
        ax.set_yticks([0, 1])
        ax.yaxis.set_ticklabels([0, 1])
        
        ax.plot([t[t_indexes][0], t[t_indexes][-1]],[0,0], color= 'black', linewidth=0.25)
        for i in range(len(sel)):
            ax.plot([t[t_indexes][0], t[t_indexes][-1]],[i+1,i+1], color= 'black', linewidth=0.25)
            
        if show_right_ax_labels:
            ax2 = ax.twinx()
            ax2.plot([t[t_indexes][0], t[t_indexes][-1]], [0, len(sel)], alpha = 0)
            ax2.set_ylabel(r'oscilator index: $i$')
        
            if sel == range(n):
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                a=np.array(ax2.get_yticks().tolist())
                a = a[ np.intersect1d(np.where(a>=0)[0], np.where(a< len(sel))[0])]
                ax2.set_yticks(a + 0.5)

                a_labels = [int(a[i]) for i in range(len(a))]
                ax2.set_yticklabels(a_labels)
            else:
                a = [ 0.5 + i for i in range(len(sel))]
                ax2.set_yticks(a)

                a_labels = [sel[i] for i in range(len(sel))]
                ax2.set_yticklabels(a_labels)
    else:
        ax.plot([0, 1], [0, 1], alpha = 0)
        
    for i in range(len(sel)):
        ax.plot(t[t_indexes], shift * i + mod2pi(theta[t_indexes, sel[i]] - np.unwrap(psi[t_indexes]) - np.pi)/(2*np.pi),
                color = get_color_from_list(sel_color[i], color),
                linewidth=0.5)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$(\theta_i - \psi - \pi)/2\pi$')

    return ax


def generate_array_density_histogram(theta, psi, i, bins=51, s=0.2):
    aux_theta = np.mod(chop(theta[:, i] - psi - np.pi, s, 1), 2 * np.pi)
    results, edges = np.histogram(aux_theta, bins=np.linspace(0, 2 * np.pi, bins), normed=True)
    binWidth = edges[1] - edges[0]

    array = results * binWidth

    return array


def plot_phase_minus_psi_histogram(theta, psi, bins=51, shift_axis=False,
                                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'],
                                   show_right_ax_labels = True,
                                   sel=None,
                                   ax=None):
    if ax is None:
        ax = plt.gca()

    n = np.shape(theta)[1]

    if sel is None:
        sel_color = range(n)
        sel = range(n)
    else:
        if color == plt.rcParams['axes.prop_cycle'].by_key()['color']:
            sel_color = range(len(sel))
        else:
            sel_color = sel

    shift = 0
    if shift_axis:
        shift = 1
        ax.set_yticks([0, 1])
        ax.yaxis.set_ticklabels([0, 1])
        ax.plot([0, 1], [0, len(sel)], alpha=0)

        if show_right_ax_labels:
            ax2 = ax.twinx()
            ax2.plot([0, 1], [0, len(sel)], alpha=0)
            ax2.set_ylabel(r'oscilator index: $i$')

            if sel == range(n):
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                a = np.array(ax2.get_yticks().tolist())
                a = a[np.intersect1d(np.where(a >= 0)[0], np.where(a < len(sel))[0])]
                ax2.set_yticks(a + 0.5)

                a_labels = [int(a[i]) for i in range(len(a))]
                ax2.set_yticklabels(a_labels)
            else:
                a = [0.5 + i for i in range(len(sel))]
                ax2.set_yticks(a)

                a_labels = [sel[i] for i in range(len(sel))]
                ax2.set_yticklabels(a_labels)
    else:
        ax.plot([0, 1], [0, 1], alpha=0)

    for i in range(len(sel)):
        array = generate_array_density_histogram(theta, psi, sel[i], bins=bins + 1)

        for j in range(bins):
            if array[j] > 0:
                ax.add_patch(patches.Rectangle(
                    (j / bins, shift * i),  # (x,y)
                    1 / bins,  # width
                    max(array[j], 0.01),  # height
                    linewidth=0.25,
                    alpha=1 if shift_axis else 0.8,
                    fill=True,
                    zorder=10,
                    edgecolor=get_color_from_list(sel_color[i], color) if shift_axis else 'black',
                    facecolor=get_color_from_list(sel_color[i], color)))

    if shift_axis:
        ax.plot([0, 0.03], [0, 0], color='black', linewidth=0.25, zorder=1)
        ax.plot([0.97, 1], [0, 0], color='black', linewidth=0.25, zorder=1)
        for i in range(len(sel)):
            ax.plot([0, 0.03], [i + 1, i + 1], color='black', linewidth=0.25, zorder=1)
            ax.plot([0.97, 1], [i + 1, i + 1], color='black', linewidth=0.25, zorder=1)

    ax.set_ylabel(r'density')
    ax.set_xlabel(r'$(\theta_i - \psi - \pi)/2\pi$')

    return ax


def plot_r_psi(t, r, psi, tlim=None, ax=None):

    if ax is None:
        ax = plt.gca()

    if tlim != None:
        indexes = np.intersect1d(np.where(t>=tlim[0])[0], np.where(t<=tlim[1])[0])
    else:
        indexes = range(len(t))

    ax.plot(t[indexes], mod2pi(np.unwrap(psi[indexes])) / (2 * np.pi), color='black', linestyle='--', label=r'$\psi$', linewidth=0.5)
    ax.plot(t[indexes], r[indexes], color='blue', label=r'$R$', linewidth=0.5)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\psi/2\pi, R$')

    return ax


cmap_density = LinearSegmentedColormap.from_list("", ["orange","#009000","black"])


def plot_r_exp_psi(t, r, psi, tlim=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if tlim != None:
        indexes = np.intersect1d(np.where(t>=tlim[0])[0], np.where(t<=tlim[1])[0])
    else:
        indexes = range(len(t))
        
    coords = [[r[i] * np.cos(psi[i]), r[i] * np.sin(psi[i])] for i in indexes]

    x, y = zip(*coords)
    points = np.array([x, y]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap_density,
                        norm=plt.Normalize(0, t[indexes[-1]] - t[indexes[0]] ))
    lc.set_array(t)
    lc.set_linewidth(0.5)

    ax.add_collection(lc)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax.set_xlabel(r'Real($R$ exp $i \psi$)')
    ax.set_ylabel(r'Im($R$ exp $i \psi$)')
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap_density, norm=plt.Normalize(vmin=t[indexes[0]], vmax=t[indexes[-1]]))
    sm._A = []
    cb1 = plt.colorbar(sm)
    cb1.set_label(r't')
    
    return ax