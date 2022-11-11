"""
    Filter the key points by Sift,
    and Select the first N points with highest response (DoG Response)
        DoG: Difference of Gaussian
"""
# import torch
import numpy as np
import cv2
import os
# from tutils import os.path.join
import yaml
# from utils import visualize
# from trans_utils import draw_landmarks

def sift_filterring():
    config = {
        'debug_files': './imgs_for_debug/',
        'sift_save_pth': './tmp_features/',
        'sift_visuals': './tmp_visuals/',
        'dataset_pth': '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData',
        # 'sift_save_pth': './runs/sift_landmarks3/RawImage/TrainingData/lm',
        # 'sift_visuals': './runs/sift_landmarks3/RawImage/TrainingData/visuals/',
        'size': [384, 384],
        'runs_dir': "./runs/",
        'tag': 'sift_landmarks',
    }

    # img_pths = [x.path for x in os.scandir(config['dataset_pth']) if x.name.endswith('.bmp')]
    img_pths = [x.path for x in os.scandir(config['debug_files']) if x.name.endswith('.jpg')]
    img_pths.sort()
    assert len(img_pths) > 0
    all_landmarks = []
    all_responses = []
    for img_pth in img_pths:
        img = cv2.imread(img_pth)
        img = cv2.resize(img, tuple(config['size']))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        # kp = sift.detect(gray, None)
        kp, des = sift.detectAndCompute(gray, None)
        import ipdb; ipdb.set_trace()
        landmark_list = []
        response_list = []
        for kpi in kp:
            response_list.append(kpi.response)
            landmark_list.append(list(kpi.pt))
        response_list = np.array(response_list)
        rank = np.argsort(response_list)[::-1]
        response_list = response_list[rank]
        landmark_list = np.array(landmark_list)
        landmark_list = landmark_list[rank]
        lm = landmark_list

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

        # print(f"img {img_pth[-7:-4]} R:", response_list)
        all_landmarks.append(landmark_list)
        all_responses.append(response_list)

        if not os.path.isdir(config['sift_visuals']):
            os.makedirs(config['sift_visuals'])
            print("make dir 1")
        if not os.path.isdir(config['sift_save_pth']):
            os.makedirs(config['sift_save_pth'])
            print("make dir 2")

        img = cv2.drawKeypoints(gray, kp, img)
        cv2.imwrite(os.path.join(config['sift_visuals'], f'clean_{img_pth[-7:-4]}.jpg'), img)
        img = cv2.drawKeypoints(gray, kp[:100], img)
        cv2.imwrite(os.path.join(config['sift_visuals'], f'clean_first100_{img_pth[-7:-4]}.jpg'), img)
        # draw_landmarks(img, landmark_list)
        # import ipdb; ipdb.set_trace()

        np.save(os.path.join(config['sift_save_pth'], f'sift_landmarks_{img_pth[-7:-4]}.npy'), np.array(all_landmarks))
        np.save(os.path.join(config['sift_save_pth'], f'sift_responses_{img_pth[-7:-4]}.npy'), np.array(all_responses))
        print(f"Processed ", img_pth, end="\r")

    with open(os.path.join(config['sift_save_pth'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print("Over!")

def review():
    # data = np.load('./runs/sift_landmarks/RawImage/TrainingData/sift_landmarks.npy', allow_pickle=True)
    # first_100_list = []
    # for i in range(len(data)):
    #     first_100_list.append(data[i][:100, :])
    # first_100_list = np.array(first_100_list)
    # np.save('./runs/sift_landmarks/RawImage/TrainingData/sift_landmarks_100.npy', first_100_list)
    data = np.load('/home1/quanquan/code/landmark/code/stat_analysis/runs/sift_landmarks2/RawImage/TrainingData/lm/sift_landmarks_001.npy', allow_pickle=True)
    data = data[0]
    print(data.shape)
    import ipdb;ipdb.set_trace()



if __name__ == '__main__':
    sift_filterring()
    # review()