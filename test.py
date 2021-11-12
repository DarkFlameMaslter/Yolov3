import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import torch

#program
cat_img = cv.imread("./data/meo.jpg")

torch.nn.Conv2d(cat_img,cat_img,3,3,0,)


plt.imshow(cv.cvtColor(cat_img, cv.COLOR_RGB2BGR))
plt.show()
plt.close(all)

input('Wait for key press!')
