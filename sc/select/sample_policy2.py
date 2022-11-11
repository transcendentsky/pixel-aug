"""
    Greedy policy
"""
from multiprocessing.sharedctypes import Value
import numpy as np

def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()

def search_new_temp(search_num=1, finded_template=[], matrix=None):
    
    # if isinstance(finded_template, list):
    #     finded_template = np.array(finded_template)
    # total_list = np.array([i for i in range(150)])
    # total_list = [i for i in range(150)]

    rest_list = [i for i in range(150) if i not in finded_template]
    # import ipdb; ipdb.set_trace()
    best_fit = 0
    best_temp = None
    if search_num == 1:
        for ti in rest_list:
            tt = [ti] + finded_template
            fit = _fitness(tt, matrix=matrix)
            if fit > best_fit:
                best_fit = fit
                best_temp = ti
        final_list = [best_temp] + finded_template
    elif search_num == 2:
        pass
    else:
        raise ValueError
    return final_list, best_fit


def greedy1():
    matrix = np.load("data_max_list_all.npy")
    num = 10
    template_list = []
    search_num = 1
    searched_num = 0
    for i in range(999):
        if searched_num >= num:
            break
        new_temps, fit = search_new_temp(search_num=search_num, finded_template=template_list, matrix=matrix)
        searched_num += search_num
        template_list = new_temps
        print(template_list, fit)
    return template_list


if __name__ == '__main__':
    from tutils import timer
    matrix = np.load("data_max_list_all.npy")
    # tlist = search_new_temp(finded_template=[], matrix=matrix)
    timer1 = timer()
    tlist = greedy1()
    tt = timer1()
    print("List and Time")
    print(tlist, tt)