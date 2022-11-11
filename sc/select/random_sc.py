"""
    Random indices for comparison
"""

import numpy as np
from tutils import trans_args, trans_init, save_script, dump_yaml, tfilename
import argparse
from tutils import CSVLogger, time_str
import subprocess



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


def random_select(logger, config, args):
    """  Random choose and get fitness  """
    matrix = np.load("../stat_analysis/data_max_list_all.npy")
    best_idx = None
    worst_fit = 999
    num = args.num
    threshold = 0.4
    save_idx_list = []
    save_fit_list = []

    logger.info(f"select num: {num}")
    for i in range(100):
        gene = np.random.choice(np.arange(1, 150), num, replace=False)
        fit = _fitness(gene, matrix=matrix)
        # if fit > threshold:
        # logger.info(f"[*] Greater than Threshold {threshold}, fit: {fit}, idx {gene}")
        # save_fit_list.append(fit)
        # gene.sort()
        # save_idx_list.append(gene)
        if fit < worst_fit:
            worst_fit = fit
            gene.sort()
            best_idx = gene
            best_idx_str = ''
            for idx in best_idx:
                best_idx_str += f"{idx}, "
            logger.info(f"[*] Iter {i}; Got new best! best fit:{worst_fit}; idx {best_idx_str}")
    return best_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/select/select.yaml")
    parser.add_argument('--func', default='random_select')
    parser.add_argument('--num', type=int, default=15)
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="a+")
    num_list  = [15, 30, 45, 60, 75]
    # num_list2 = [22, 37, 52, 67, 82]
    num_list2 = []

    llist = num_list + num_list2
    for n in llist:
        args.num = n
        worst_ids = eval(args.func)(logger, config, args)

        csvlogger.record({
            'num': n,
            'best_ids': worst_ids,
            'time': time_str(),
        })
        inds_str = ",".join([str(s) for s in worst_ids])
        p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES=5 python -m sc.baseline.baseline_part "
                             f"--tag worst_num_{args.num} --indices {inds_str}", shell=True)
        p.wait()