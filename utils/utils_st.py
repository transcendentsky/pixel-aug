import numpy as np
import torch
from multiprocessing import Process, Queue
from torchvision.transforms import ToPILImage


def voting_channel(k, heatmap, regression_y, regression_x, \
                   Radius, spots_y, spots_x, queue, num_candi):
    n, c, h, w = heatmap.shape

    score_map = np.zeros([h, w], dtype=int)
    for i in range(num_candi):
        vote_x = regression_x[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_y = regression_y[0, k, spots_y[0, k, i], spots_x[0, k, i]]
        vote_x = spots_x[0, k, i] + int(vote_x * Radius)
        vote_y = spots_y[0, k, i] + int(vote_y * Radius)
        if vote_x < 0 or vote_x >= w or vote_y < 0 or vote_y >= h:
            # Outbounds
            continue
        score_map[vote_y, vote_x] += 1
    score_map = score_map.reshape(-1)
    queue.put([k, score_map.argmax(), score_map.max()])


def voting(heatmap, regression_y, regression_x, Radius, get_voting=False):
    # n = batchsize = 1
    heatmap = heatmap.cpu()
    regression_x, regression_y = regression_x.cpu(), regression_y.cpu()
    n, c, h, w = heatmap.shape
    assert (n == 1)

    num_candi = int(3.14 * Radius * Radius)

    # Collect top num_candi points
    score_map = torch.zeros(n, c, h, w, dtype=torch.int16)
    spots_heat, spots = heatmap.view(n, c, -1).topk(dim=-1, \
                                                    k=num_candi)
    spots_y = spots // w
    spots_x = spots % w

    process_list = list()
    queue = Queue()
    for k in range(c):
        process = Process(target=voting_channel, args=(k, heatmap, \
                                                       regression_y, regression_x, Radius, spots_y, spots_x, queue,
                                                       num_candi))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()

    landmark = np.zeros([c], dtype=int)
    votings = np.zeros([c], dtype=int)
    for i in range(c):
        out = queue.get()
        landmark[out[0]] = out[1]
        votings[out[0]] = out[2]

        # This is for guassian mask
        # landmark[i] = heatmap[0][i].view(-1).max(0)[1]
    landmark_y = landmark / w
    landmark_x = landmark % w
    if get_voting: return [landmark_y.astype(int), landmark_x], votings
    return [landmark_y.astype(int), landmark_x]



