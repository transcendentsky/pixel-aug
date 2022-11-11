"""
    paste from deepEMD
"""

import cv2
import torch
import torch.nn.functional as F
from einops import rearrange


def emd_for_cc2d(raw_fea_list, crop_fea_list, raw_loc, chosen_loc):
    b = raw_loc.shape[0]
    # raw_vectors = []
    loss = []
    # print("debug in emd_for_cc2d", raw_loc, chosen_loc)
    for ii in range(b): # per level
        raw_vectors = torch.stack([raw_fea_list[j][[ii], :, raw_loc[ii][0]//2**(5-j), raw_loc[ii][1]//2**(5-j)] for j in range(5)]).squeeze()
        tmp_vectors = torch.stack([crop_fea_list[j][[ii], :, chosen_loc[ii][0]//2**(5-j), chosen_loc[ii][1]//2**(5-j)] for j in range(5)]).squeeze()
        proto = rearrange(raw_vectors, "l c -> 1 c l 1")
        query = rearrange(tmp_vectors, "l c -> 1 c l 1")
        # import ipdb; ipdb.set_trace()
        logits = emd_forward_1shot(proto, query)
        print(logits)
        loss.append(logits)
    return - torch.stack(loss).mean()


def emd_forward_1shot(proto, query):
    """
    :param proto:  (way, c, n, b) eg. (5,640,9,1)
    :param query:  (way, c, n, b)
    :return:
    """
    # assert
    proto = proto.squeeze(0)

    weight_1 = get_weight_vector(query, proto)
    if len(proto.shape) != 4:
        proto = proto.view(1, 16, 5, 1)
        query = query.view(1, 16, 5, 1)
    weight_2 = get_weight_vector(proto, query)

    proto = normalize_feature(proto)
    query = normalize_feature(query)

    similarity_map = get_similiarity_map(proto, query)
    logits = get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
    return logits


def get_weight_vector(A, B):
    M = A.shape[0]
    N = B.shape[0]

    # import ipdb; ipdb.set_trace()
    B = F.adaptive_avg_pool2d(B, [1, 1])
    B = B.repeat(1, 1, A.shape[2], A.shape[3])

    A = A.unsqueeze(1)
    B = B.unsqueeze(0)

    A = A.repeat(1, N, 1, 1, 1)
    B = B.repeat(M, 1, 1, 1, 1)

    combination = (A * B).sum(2)
    combination = combination.view(M, N, -1)
    combination = F.relu(combination) + 1e-3
    return combination


def normalize_feature(x, norm="center"):
    if norm == 'center':
        x = x - x.mean(1).unsqueeze(1)
        return x
    else:
        return x


def get_similiarity_map(proto, query, metric="cosine"):
    way = proto.shape[0]
    num_query = query.shape[0]
    query = query.view(query.shape[0], query.shape[1], -1)
    proto = proto.view(proto.shape[0], proto.shape[1], -1)

    proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    query = query.unsqueeze(1).repeat([1, way, 1, 1])
    proto = proto.permute(0, 1, 3, 2)
    query = query.permute(0, 1, 3, 2)
    feature_size = proto.shape[-2]

    if metric == 'cosine':
        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = F.cosine_similarity(proto, query, dim=-1)
    elif metric == 'l2':
        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = (proto - query).pow(2).sum(-1)
        similarity_map = 1 - similarity_map

    return similarity_map


def get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv'):
    num_query = similarity_map.shape[0]
    num_proto = similarity_map.shape[1]
    num_node = weight_1.shape[-1]
    temperature = 12.5

    # if solver == 'opencv':  # use openCV solver
    for i in range(num_query):
        for j in range(num_proto):
            _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

            similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

    temperature = (temperature / num_node)
    logitis = similarity_map.sum(-1).sum(-1) * temperature
    return logitis


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow

'''
python 学习 OpenCV
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_transform():
    img = cv2.imread('1cun2.jpg', 0)
    img1 = np.array(img/3, dtype='uint8')
    img2 = np.array(img/3+85, dtype='uint8')
    img3 = np.array(img/3+170, dtype='uint8')

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255])  # 获得灰度直方图
    hist1 = cv2.normalize(hist1, 0, 1)  # 直方图归一化,不归一化也行
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255])
    hist2 = cv2.normalize(hist2, 0, 1)
    hist3 = cv2.calcHist([img3], [0], None, [256], [0, 255])
    hist3 = cv2.normalize(hist3, 0, 1)
    retval12 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # 相关性比较
    retval13 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_CORREL)
    chisqr12 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)  # 卡方比较
    chisqr13 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_CHISQR)
    inters12 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)  # 直方图相交比较
    inters13 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_INTERSECT)
    bhatta12 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)  # 巴氏距离比较
    bhatta13 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_BHATTACHARYYA)

    def hist2signature(hist):
        signature = np.zeros(shape=(hist.shape[0] * hist.shape[1], 2), dtype=np.float32)
        for h in range(hist.shape[0]):
            idx = h
            signature[idx][0] = hist[h]
            signature[idx][1] = h
        return signature
    signature1 = hist2signature(hist1)  # EMD函数不能直接输入hist，需要转化为（值，索引）数组
    signature2 = hist2signature(hist2)
    signature3 = hist2signature(hist3)
    print(signature1)
    retvalEMD12, lowerBound12, flow12 = cv2.EMD(signature1, signature2, cv2.DIST_L2)  # EMD比较
    retvalEMD13, lowerBound13, flow13 = cv2.EMD(signature1, signature3, cv2.DIST_L2)

    print('retval12', retval12)
    print('retval13', retval13)
    print('chisqr12', chisqr12)
    print('chisqr13', chisqr13)
    print('inters12', inters12)
    print('inters13', inters13)
    print('bhatta12', bhatta12)
    print('bhatta13', bhatta13)
    print('retvalEMD12', retvalEMD12, '  lowerBound12', lowerBound12)
    print('retvalEMD13', retvalEMD13, '  lowerBound13', lowerBound13)

    plt.figure(figsize=(14, 8))
    plt.subplot(231)
    plt.imshow(img1, 'gray', vmin=0, vmax=255)
    plt.subplot(232)
    plt.imshow(img2, 'gray', vmin=0, vmax=255)
    plt.subplot(233)
    plt.imshow(img3, 'gray', vmin=0, vmax=255)
    plt.subplot(234)
    plt.plot(hist1, color='c')
    plt.subplot(235)
    plt.plot(hist2, color='c')
    plt.subplot(236)
    plt.plot(hist3, color='c')
    plt.show()

    plt.figure(figsize=(14, 8))
    plt.plot(hist1, 'r')
    plt.plot(hist2, 'g')
    plt.plot(hist3, 'b')
    plt.show()
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    img_transform()