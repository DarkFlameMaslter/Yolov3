#this is the impliment for yolov, the core

import torch
import torch.nn as nn

"""
this is the test string
tuple? is structured by (filter, kernel_size, stride)

every conv is a same convolution

list note:
B: is sign 4 a residual block followed by the number of repeats
S for the scale prediction block and caculate the yolo loss
U for upsampling the feature map and concatenating with a previous layer

"""


config = [
    (32, 3, 1),  #(filter, size x size, repeat)
    (64, 3, 2),
    ["B", 1],    #one time residual block
    (128, 3, 2), #single convolutional layer
    ["B", 2],    #two time residual block...
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",        #82
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",        #94
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",        #106
]
