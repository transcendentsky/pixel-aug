"""
    Draw 3d maps for variations of augmentations
"""


import numpy as np
# from tutils import tfilename
import os
import cv2
from sklearn.metrics import mutual_info_score
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns


def draw_surface():
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    Z = [[3, 3, 3., 3],
         [3, 4, 3.5, 3],
         [3, 3, 3., 3],
         [3, 3, 3, 3],]

    X, Y = np.meshgrid(x, y)
    Z = np.array(Z)
    print(X)
    print(Y)

    fig = plt.figure()
    ax = Axes3D(fig)

    # 绘制平面  z=3
    ax.plot_surface(X, Y, Z=Z)

    # 设置x,y,z标签
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("./tmp/f2d.png")


def draw_line_plot():
    # Sequences
    x = np.array(range(8))
    # linear_sequence = [1, 2, 3, 4, 5, 6, 7, 10, 15, 20]
    # exponential_sequence = np.exp(np.linspace(0, 10, 10))
    linear_sequence = [0.94, 0.93, 0.922, 0.916, 0.90, 0.84, 0.7, 0.4]
    exponential_sequence = [3.0, 3.0, 2.8, 2.61, 2.5, 2.64, 2.78, 3.0]
    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    sns.lineplot(ax=ax, x=x, y=linear_sequence, color='red')
    # ax.plot(linear_sequence, color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_ylabel("VI")
    ax.set_xlabel("Augmentation intensity")
    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()

    # Plot exponential sequence, set scale to logarithmic and change tick color
    sns.lineplot(ax=ax2, x=x, y=exponential_sequence, color='green')
    # ax2.plot(exponential_sequence, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel("MRE")
    plt.show()


if __name__ == '__main__':
    # draw_surface()
    draw_line_plot()