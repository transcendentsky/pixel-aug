import numpy as np
from einops import rearrange

# x = np.arange(0,5)
# y = np.arange(0,5)
# xy = np.meshgrid(x, y)
# xy = np.array(xy)
# xy = rearrange(xy, "c w h -> h w c")
# print(xy)
prob = np.zeros((384,384))
prob[0, 172] = 1
prob[250,250] = 1
prob[300,125] = 1
prob /= prob.sum()
# ---
xy = np.arange(0, 384*384)
prob = rearrange(prob, "h w -> (h w)")

for i in range(10000):
    loc = np.random.choice(a=xy, size=1, replace=False, p=prob)[0]
    loc = [loc // 384, loc % 384]
    print(i, loc, end='\r')
# import ipdb; ipdb.set_trace()