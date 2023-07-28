# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:51:12 2023

@author: richie bao
"""
import matplotlib as mpl
from matplotlib import pyplot as plt, colors

def plot_style_axis_A(ax):
    # Move the left and bottom spines to x = 0 and y = 0, respectively.
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    # Hide the top and right spines.
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    
def vals4color_cmap(vals,cmap='hot',vmin=0, vmax=50):    
    cmap = mpl.colormaps[cmap]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    color=cmap(norm(vals))
    return color    