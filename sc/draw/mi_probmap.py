import numpy as np
from tutils import trans_init, trans_args, dump_yaml, tfilename, CSVLogger
import argparse
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt



def seaborn_scatter(x, y, fname="ttest.pdf", xlabel="x", ylabel="y", color="blue"):
    sns.set_theme(style="whitegrid", font='Times New Roman', font_scale=1.2)
    # fig = sns.scatterplot(x=x, y=y, color=color)
    fig = sns.regplot(x=x, y=y, color=color, line_kws={'color': "cyan", 'alpha': 0.5})
    fig.text(0.54,4.75, "cc=-0.675", horizontalalignment='left', size='medium', color='black', weight='semibold')
    scatter_fig = fig.get_figure()
    fig.set(xlabel=xlabel, ylabel=ylabel)
    scatter_fig.savefig(fname, dpi=400)
    print("Save to: ", fname)
