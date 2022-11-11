"""
    Edge detection for comparison
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread("026.bmp", cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (384,384))
edges = cv2.Canny(im, 50, 200)

plt.imshow(edges, cmap='gray')
plt.show()
plt.imsave("026_edge.png", edges, cmap='gray')