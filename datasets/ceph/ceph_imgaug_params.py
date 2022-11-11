import numpy as np

cj_br_floor = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cj_br_ceil = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cj_ct_floor = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cj_ct_ceil = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

res_list = []
for i in range(19):
    res_d = {
        "index": i,
        "cj_br_ceil": cj_br_ceil[i],
        "cj_ct_ceil": cj_ct_ceil[i],
        "cj_br_floor": cj_br_floor[i],
        "cj_ct_floor": cj_ct_floor[i],
    }
    res_list.append(res_d)
    print(res_d)
np.save("./cache/optimal_brct_2.npy", res_list)