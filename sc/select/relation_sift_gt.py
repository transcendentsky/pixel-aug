import numpy as np
from tutils import trans_init, trans_args, dump_yaml, tfilename, CSVLogger
import argparse
import seaborn as sns



import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
def draw_scatter(points, points2, fname="ttest.png", c="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(7,5))
    # fig, ax = plt.subplots
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c) # , label="???")
    fontsize_ticks = 16
    fontsize_label = 18
    plt.xlabel(xlabel, fontsize=fontsize_label, labelpad=8)
    plt.ylabel(ylabel, fontsize=fontsize_label, labelpad=8)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(fname)
    print("Save to: ", fname)
    plt.close()


def seaborn_scatter(x, y, fname="ttest.pdf", xlabel="x", ylabel="y", color="blue"):

    sns.set_theme(style="whitegrid", font='Times New Roman', font_scale=1.2)
    # fig = sns.scatterplot(x=x, y=y, color=color)
    fig = sns.regplot(x=x, y=y, color=color, line_kws={'color': "cyan", 'alpha': 0.5})
    fig.text(0.54,4.75, "cc=-0.675", horizontalalignment='left', size='medium', color='black', weight='semibold')
    scatter_fig = fig.get_figure()
    fig.set(xlabel=xlabel, ylabel=ylabel)
    scatter_fig.savefig(fname, dpi=400)
    print("Save to: ", fname)


def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()


def tmp(logger, config):
    # sift_tag = 'sift2'
    data_list = []
    for idx in range(150):
        # "/home1/quanquan/code/landmark/code/runs-ana/sift1/max_list/data_max_list_oneshot_0.npy"
        print(f"****  idx: {idx} ", end=" \r")
        # This is the
        # pth = f"/home1/quanquan/code/landmark/code/runs/select/" \
        #       f"maxsim_sift/pixelcontra/max_list/data_max_list_oneshot_{idx}.npy"
        pth = f"/home1/quanquan/code/landmark/code/runs-ana/sift2/max_list/data_max_list_oneshot_{idx}.npy"
        data = np.load(pth)
        data = data[:, :, -1]
        data_list.append(data)
        # import ipdb; ipdb.set_trace()
    data_np = np.array(data_list)
    print("data np .shape", data_np.shape)
    np.save(tfilename(config['base']['runs_dir'], f"data_max_list_all.npy"), data_np)


def _tmp_load_data(n, max_list_all_sift):
    maxsim_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_1/conf_n{n}_list.npy"
    ids_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_1/ids_n{n}_list.npy"
    conf_lm = np.load(maxsim_list)
    ids_list = np.load(ids_list)

    conf_lm = np.array(conf_lm)
    conf_sift = [_fitness(x, max_list_all_sift) for x in ids_list]

    cc = np.corrcoef(conf_lm, conf_sift)
    return conf_lm, conf_sift, cc

# def draw_4_figs(logger, config):
#     llist = [1,2,3,4]
#     max_list_all_sift = np.load(tfilename(config['base']['runs_dir'], f"data_max_list_all.npy"))
#
#     fig, axes = plt.subplots(2,2)
#     for i in llist:
#         conf_lm, conf_sift, cc = _tmp_load_data(i, max_list_all_sift)
#         axes[0].scatterplot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../runs/relation/")
    parser.add_argument('--experiment', default="relation")
    parser.add_argument('--ref', type=int, default=1)
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)

    tmp(logger, config)

    n = args.ref
    maxsim_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_1/conf_n{n}_list.npy"
    ids_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_1/ids_n{n}_list.npy"

    max_list_all_sift = np.load(tfilename(config['base']['runs_dir'], f"data_max_list_all.npy"))
    # maxsim_list_sift = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_sift/max_list_sift_n{n}.npy"

    conf_lm = np.load(maxsim_list)
    # maxsim_list_sift = np.load(maxsim_list_sift)
    ids_list = np.load(ids_list)
    # import ipdb; ipdb.set_trace()

    conf_sift = [_fitness(x, max_list_all_sift) for x in ids_list]

    conf_lm = np.array(conf_lm)
    conf_sift = np.array(conf_sift)


    cc = np.corrcoef(conf_lm, conf_sift)
    logger.info(f"ref: {n};  cc: {cc}")
    data = np.stack([conf_sift, conf_lm], axis=-1)
    # csvlogger = CSVLogger(tfilename(config['base']['runs_dir'], "record"))
    # for i in range(data.shape[0]):
    #     d = {j: data[i][j] for j in range(data.shape[1]) }
    #     csvlogger.record(d)

    seaborn_scatter(conf_lm, conf_sift, fname=tfilename(config['base']['runs_dir'], f"relation_gt_sift_n{n}.png"), xlabel="key points of interest", ylabel="potential key points")
    seaborn_scatter(conf_lm, conf_sift, fname=tfilename(config['base']['runs_dir'], f"relation_gt_sift_n{n}.pdf"), xlabel="key points of interest", ylabel="potential key points")
    # draw_scatter(conf_lm, conf_sift, fname=tfilename(config['base']['runs_dir'], f"relation_gt_sift_n{n}.png"), xlabel="key points of interest", ylabel="potential key points")
    # draw_scatter(conf_lm, conf_sift, fname=tfilename(config['base']['runs_dir'], f"relation_gt_sift_n{n}.pdf"), xlabel="key points of interest", ylabel="potential key points")

    # import ipdb; ipdb.set_trace()

