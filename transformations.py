#!/usr/bin/env python3
"""module which contains transformations function"""
import numpy as np
import matplotlib.pyplot as plt


def transformations(*f, **coordinates):
    """
    - f: list of functions
    - coordinates: contains coordinates
    Returns: A dictionary comprenhension containing
             function name, transformation pairs
    """
    try:
        return {name: (x, np.array([f[i](yj) if yj else 0 for yj in y])) for i, (name, (x, y)) in enumerate(coordinates.items())}
    except Exception as e:
        raise Exception("Usage: both vectors of functions and coordinates must have the same length")

if __name__ == '__main__':
    functions = [lambda x : 1 / (1 + np.exp(-x)) if x else 0,
                 lambda x: 1 / x]
    
    coordinates = {'sigmoid': (np.arange(-6, 6, .1), np.arange(-6, 6, .1).tolist()),
                   'hyperbola': (np.arange(-100, 100), np.arange(-100, 100).tolist())}
    to_plot = transformations(*functions, **coordinates)

    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace=.3)
    for i, key in enumerate(to_plot):
        x, y = to_plot.get(key)
        axs[i].set_title(key)
        if key == 'hyperbola':
            y = np.ma.masked_equal(y, 0)
        axs[i].plot(x, y, linewidth=.8)
        axs[i].axhline(y=(y[-1] - y[0]) / 2,
                    color='red', linestyle='--',
                    linewidth=.6)
    plt.show()
