"""
Camera calibrator with trivial way
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

import utils 
# 加载棋盘格图片
image = cv2.imread('./Final project/chessboard.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
# utils.get_corner_pixel(gray)
utils.get_corner_world()