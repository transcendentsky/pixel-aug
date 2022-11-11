"""
    Filter the key points by Sift,
    and Select the first N points with highest response (DoG Response)
        DoG: Difference of Gaussian
"""
import torch
import numpy as np
import cv2
import os
from tutils import tfilename
import yaml
from tutils import trans_init, trans_args, save_script, dump_yaml, CSVLogger
import argparse
from tutils import print_dict

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt




def sift_filterring(logger, config):
    img_pths = [x.path for x in os.scandir(tfilename(config['dataset']['pth'], 'RawImage/TrainingData')) if x.name.endswith('.bmp')]
    img_pths.sort()
    assert len(img_pths) > 0
    # all_landmarks = []
    # all_responses = []
    # all_des = []
    for img_pth in img_pths:
        img = cv2.imread(img_pth)
        img = cv2.resize(img, tuple((384, 384)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        # kp = sift.detect(gray, None)
        kp, des = sift.detectAndCompute(gray, None)
        # all_des.append(des)
        landmark_list = []
        response_list = []
        for i, kpi in enumerate(kp):
            response_list.append(kpi.response)
            landmark_list.append(list(kpi.pt))
        response_list = np.array(response_list)
        rank = np.argsort(response_list)[::-1]
        response_list = response_list[rank]
        landmark_list = np.array(landmark_list)
        landmark_list = landmark_list[rank]
        lm = landmark_list
        des_list = des[rank]

        # Remove the points near borders
        kp = np.array(kp)[rank]
        def check(lm):
            if (lm[0] > 38.4 and lm[0] < 345.6 and lm[1] > 38.4 and lm[1] < 345.6):
                return True
            else: return False

        ind = [check(lmi) for lmi in lm]
        kp = kp[ind]
        landmark_list = landmark_list[ind]
        response_list = response_list[ind]
        des_list = des_list[ind]

        assert len(des_list) == len(landmark_list) and len(des_list) == len(response_list)
        # print(f"img {img_pth[-7:-4]} R:", response_list)
        # all_landmarks.append(landmark_list)
        # all_responses.append(response_list)

        # img = cv2.drawKeypoints(gray, kp, img)
        # cv2.imwrite(tfilename(config['sift_visuals'], f'clean_{img_pth[-7:-4]}.jpg'), img)
        # img = cv2.drawKeypoints(gray, kp[:100], img)
        # cv2.imwrite(tfilename(config['sift_visuals'], f'clean_first100_{img_pth[-7:-4]}.jpg'), img)
        # draw_landmarks(img, landmark_list)
        import ipdb; ipdb.set_trace()

        np.save(tfilename(config['base']['runs_dir'], f'lm/sift_landmarks_{img_pth[-7:-4]}.npy'), np.array(landmark_list))
        np.save(tfilename(config['base']['runs_dir'], f'lm/sift_responses_{img_pth[-7:-4]}.npy'), np.array(response_list))
        np.save(tfilename(config['base']['runs_dir'], f'lm/sift_descriptor_{img_pth[-7:-4]}.npy'),np.array(des_list))
        print(f"Processed ", img_pth, end="\r")

    # with open(tfilename(config['base']['runs_dir'], 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f)

    print("Over!")


def sift_matching(logger , config):
    img_pths = [x.path for x in os.scandir(tfilename(config['dataset']['pth'], "RawImage", "TrainingData")) if x.name.endswith('.bmp')]
    assert len(img_pths) > 0
    img_pths.sort()
    img_pths = img_pths[:2]
    des_collect = []
    gray_collect = []
    kp_collect = []
    for img_pth in img_pths:
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (384, 384))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_collect.append(gray)
        sift = cv2.SIFT_create()
        # kp = sift.detect(gray, None)
        kp, des = sift.detectAndCompute(gray, None)
        # import ipdb; ipdb.set_trace()
        des_collect.append(des)
        kp_collect.append(kp)
    print(des_collect[0], des_collect[1])
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_collect[0], des_collect[1], k=2)

    import ipdb; ipdb.set_trace()
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    mdist = [m.distance for m,n in matches]
    ndist = [n.distance for m,n in matches]
    plt.hist(mdist, bins=20)
    plt.savefig("mdist_debug.png", alpha=0.4)
    plt.hist(ndist, bins=20)
    plt.savefig("ndist_debug.png", alpha=0.4)
    import ipdb; ipdb.set_trace()

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(gray_collect[0], kp_collect[0], gray_collect[1], kp_collect[1], good, None, flags=2)
    # plt.savefig("balabala.png")
    # plt.show()
    cv2.imwrite("balabala.png", img3)

def debug(logger, config):
    img_pths = [x.path for x in os.scandir(tfilename(config['dataset']['pth'], "RawImage", "TrainingData")) if x.name.endswith('.bmp')]
    assert len(img_pths) > 0
    img_pths.sort()
    img_pths = img_pths[:2]
    size = 192
    img1 = cv2.imread(img_pths[0])          # queryImage
    img2 = cv2.imread(img_pths[1]) # trainImage
    img1 = cv2.resize(img1, (size, size))
    img2 = cv2.resize(img2, (size, size))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(img1.shape)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    import ipdb; ipdb.set_trace()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    if True:
        # BFMatcher with default params
        import ipdb; ipdb.set_trace()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="configs/")
    parser.add_argument('--base_dir', default='../runs/sift')
    parser.add_argument("--experiment", default='sift')
    args = trans_args(parser)
    logger, config = trans_init(args)
    print_dict(config)
    save_script(config['base']['runs_dir'], __file__)
    print_dict(config)
    sift_filterring(logger, config)
