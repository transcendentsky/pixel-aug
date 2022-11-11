"""
    Draw graph of entropy percentage
    used for paper writing.

    add plot for augment intensity
"""
import os

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
import numpy as np


def draw0():
    data = pd.read_csv("entr1.csv")
    fig = plt.figure(figsize=(8,6))
    ax = sns.lineplot(x="percentage", y="mre", data=data, color="red")
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_ylabel("Mean Radial Error (MRE)")
    ax2 = ax.twinx()
    ax2 = sns.lineplot(ax=ax2, x="percentage", y="gtv", data=data, color="green")
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel("Positive Similarity Indicator (PSI)")
    ax.set_xlabel("Selection Probability of High Entropy Area (%)")
    scatter_fig = ax.get_figure()
    plt.show()
    scatter_fig.savefig("entr_select.pdf")


def draw1():
    dir_pth = "./abla_entr_data"
    palette = np.array(sns.color_palette(n_colors=5))
    fig = plt.figure(figsize=(8,6))
    for i, pth in enumerate(os.listdir(dir_pth)):
        data = pd.read_csv(os.path.join(dir_pth, pth))[30:400]
        ax = sns.lineplot(x="Step", y="Value", data=data, alpha=0.25, color=palette[i])
    dir_pth = "./abla_entr_smoothed_data"
    name_list = ["0.9", "0.7", "0.5", "0.3", "0.1"]
    for i, pth in enumerate(os.listdir(dir_pth)):
        data = pd.read_csv(os.path.join(dir_pth, pth))
        ax = sns.lineplot(x="Step", y="Value", data=data, alpha=1, color=palette[i], label=name_list[i])

    ax.set_ylabel("PSI")
    ax.set_xlabel("epoch")
    scatter_fig = ax.get_figure()
    plt.show()
    scatter_fig.savefig("entr_line.pdf")



def draw_aug0():
    data = pd.read_csv("aug1.csv")
    fig = plt.figure(figsize=(8,6))
    ax = sns.lineplot(x="percentage", y="mre", data=data, color="red")
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_ylabel("Mean Radial Error (MRE)")
    ax2 = ax.twinx()
    ax2 = sns.lineplot(ax=ax2, x="percentage", y="gtv", data=data, color="green")
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel("Positive Similarity Indicator (PSI)")
    ax.set_xlabel("Augmentation Intensity (%)")
    scatter_fig = ax.get_figure()
    plt.show()
    scatter_fig.savefig("aug_select.pdf")

def draw_aug1():
    dir_pth = "./abla_aug_data"
    palette = np.array(sns.color_palette(n_colors=7))
    fig = plt.figure(figsize=(8, 6))
    for i, pth in enumerate(os.listdir(dir_pth)):
        print(pth)
        data = pd.read_csv(os.path.join(dir_pth, pth))[25:350]
        ax = sns.lineplot(x="Step", y="Value", data=data, alpha=0.25, color=palette[i])
    dir_pth = "./abla_aug_smoothed_data"
    name_list = ["0.5", "1.2", "1.8" ,"2.4", "3.0", "4.0"]
    for i, pth in enumerate(os.listdir(dir_pth)):
        data = pd.read_csv(os.path.join(dir_pth, pth))
        ax = sns.lineplot(x="Step", y="Value", data=data, alpha=1, color=palette[i], label=name_list[i])
    ax.set_ylabel("PSI")
    ax.set_xlabel("epoch")
    scatter_fig = ax.get_figure()
    plt.show()
    scatter_fig.savefig("aug_line.pdf")

if __name__ == '__main__':
    # draw0()
    # draw1()
    # draw_aug0()
    draw_aug1()