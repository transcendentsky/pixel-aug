from PIL import Image
from tutils import tfilename
import numpy as np
import os

def get_lms(path):
    lms = []
    with open(path, "r") as f:
        num_lm = int(f.readline())
        assert num_lm == 10
        for i in range(num_lm):
            s = f.readline()
            s = s.split(" ")
            lm = [float(s[1]), float(s[0])]
            lms.append(lm)

if __name__ == '__main__':
    dir_path = "./"
    name = "03920922_03920922_x_4ZVZ4EX5_1W0LXV5Y_I0000000"
    img = Image.open(tfilename(dir_path, name + ".png")).convert("RGB")
    img_shape = np.array(img).shape[:2]  #(h, w)
    landmarks = get_lms(tfilename(dir_path, name + ".txt"))
    landmarks = [[lm[1]*img_shape[1], lm[0]*img_shape[0]] for lm in landmarks]
    import ipdb; ipdb.set_trace()