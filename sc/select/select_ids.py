import numpy as np
from tutils import trans_args, trans_init, save_script, dump_yaml, tfilename
import argparse
from tutils import CSVLogger, time_str
import subprocess

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def draw_scatter(points, points2, fname="ttest.png", c="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c, label="???")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()


def tmp(logger, config):
    # sift_tag = 'sift2'
    data_list = []
    for idx in range(150):
        # "/home1/quanquan/code/landmark/code/runs-ana/sift1/max_list/data_max_list_oneshot_0.npy"
        print(f"****  idx: {idx} ", end=" \r")
        # This is the
        # pth = f"/home1/quanquan/code/landmark/code/runs/select/" \
        #       f"maxsim_sift/pixelcontra/max_list/data_max_list_oneshot_{idx}.npy"
        # pth = f"/home1/quanquan/code/landmark/code/runs-ana/sift1/max_list/data_max_list_oneshot_{idx}.npy"
        # pth = f"/home1/quanquan/code/landmark/code/runs/select/maxsim_sift/cc2d_n73_d180/max_list/data_max_list_oneshot_{idx}.npy"
        # pth = f"/home1/quanquan/code/landmark/code/runs/select/maxsim_sift/cc2d_n50_d400/max_list/data_max_list_oneshot_{idx}.npy"
        pth = f"/home1/quanquan/code/landmark/code/runs/select/maxsim_sift/cc2d_d80/max_list/data_max_list_oneshot_{idx}.npy"
        data = np.load(pth)
        data = data[:, :, -1]
        data_list.append(data)
        # import ipdb; ipdb.set_trace()
    data_np = np.array(data_list)
    print("data np .shape", data_np.shape)
    np.save(tfilename(config['base']['runs_dir'], f"data_max_list_all.npy"), data_np)


def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()



def random_select(logger, config, args):
    """  Random choose and get fitness  """
    # matrix = np.load("../stat_analysis/data_max_list_all.npy")
    matrix = np.load(tfilename(config['base']['runs_dir'], 'data_max_list_all.npy'))
    best_idx = None
    best_fit = 0
    num = args.num
    threshold = 0.4
    save_idx_list = []
    save_fit_list = []
    save_mre_list = []
    history_best = []

    logger.info(f"select num: {num}")
    for i in range(400000):
        if num == 1:
            gene = [i]
            if i >= 150:
                break
            fit = _fitness(gene, matrix=matrix)
            # _dd = test_specific_ids(gene)
            # mre = _dd['mre']
            # save_fit_list.append(fit)
            # save_idx_list.append(i)
            # save_mre_list.append(mre)
        else:
            gene = np.random.choice(np.arange(1, 150), num, replace=False)
            fit = _fitness(gene, matrix=matrix)
        # print(f"num {i}, fitness: {fit}")
        # if fit > threshold:
            # logger.info(f"[*] Greater than Threshold {threshold}, fit: {fit}, idx {gene}")
            # save_fit_list.append(fit)
            # gene.sort()
            # save_idx_list.append(gene)
        if fit > best_fit:
            best_fit = fit
            gene.sort()
            best_idx = gene
            _dd = test_specific_ids(gene)
            print(_dd)
            # history_best.append(gene)
            best_idx_str = ''
            for idx in best_idx:
                best_idx_str += f"{idx},"
            # best_idx_str = ','.join(gene)
            logger.info(f"[*] Iter {i}; Got new best! best fit:{best_fit}; idx {best_idx_str}")

    if False and num == 1:
        rank = np.argsort(save_fit_list)
        save_idx_list = np.array(save_idx_list)[rank]
        save_fit_list = np.array(save_fit_list)[rank]
        save_mre_list = np.array(save_mre_list)[rank]

        print(save_idx_list)
        print(save_fit_list)
        draw_scatter(save_fit_list, save_mre_list, fname=tfilename(config['base']['runs_dir'], "mre_fit.png"))
        import ipdb; ipdb.set_trace()
    # save_idx_list = np.array(save_idx_list)
    # save_fit_list = np.array(save_fit_list)
    # fit_sort = np.argsort(save_fit_list)[-10:]
    # save_fit_list_10 = save_fit_list[fit_sort]
    # save_idx_list_10 = save_idx_list[fit_sort]
    # logger.info(save_fit_list_10)
    # logger.info(save_idx_list_10)
    return best_idx


if __name__ == '__main__':
    from .test_specific_ids import test_specific_ids
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/select/select.yaml")
    parser.add_argument('--func', default='random_select')
    parser.add_argument('--num', type=int, default=15)
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)

    tmp(logger, config)

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="a+")
    # num_list  = [15, 30, 45, 60, 75]
    # num_list = [1,2,3,4,5,10,15,30,45,75]
    # num_list2 = [22, 37, 52, 67, 82]
    num_list = [5] # 3,4,5
    num_list2 = []

    llist = num_list + num_list2
    for n in llist:
        args.num = n
        best_ids = eval(args.func)(logger, config, args)
        inds_str = ",".join([str(s) for s in best_ids])

        _d = test_specific_ids(best_ids)
        logger.info(_d)
        _d['num'] = n
        _d['best_ids'] = best_ids
        _d['time'] = time_str()
        csvlogger.record(_d)

        # test_specific_ids
        # p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 python -m sc.baseline.baseline_part "
        #                 f"--tag best_num_{args.num} --indices {inds_str}", shell=True)
        # p.wait()