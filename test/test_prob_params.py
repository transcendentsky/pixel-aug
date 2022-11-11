import numpy as np


thres = [1.5, 2.6, 3.4, 4.3, ]
value = [0.6198, 0.5977, 0.5412, 0.500]

thress = [1.5, 2.6, 3.4, 4.3, 5]
values = [0.7654, 0.7019, 0.7103, 0.7013, 0.7648]

temp_list = []

for i in range(4):
    for j in range(i+1, 4):
        t1, t2 = thres[i], thres[j]
        v1, v2 = value[i], value[j]

        temp = np.log(v1/v2) / np.log(t2/t1)
        temp_list.append(temp)
tau = np.array(temp_list).mean()
print(np.array(temp_list).mean())

alpha_list = []

for i in range(4):
    a = value[i] / np.exp(tau)
    alpha_list.append(a)
a_mean = np.mean(alpha_list)
print(a_mean)