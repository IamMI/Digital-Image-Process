"""
Camera calibrator with trivial way
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

import utils 

def main(image):
    M = utils.get_M(image)
    
    A = M[:, :3]  # 提取 A 矩阵 (3x3)
    b = M[:, 3]   # 提取 b 向量 (3,)

    # 分解 A 的行向量
    a1, a2, a3 = A

    # 计算 rho
    rho = 1 / np.linalg.norm(a3)

    # 计算 cx 和 cy
    cx = rho**2 * np.dot(a1, a3)
    cy = rho**2 * np.dot(a2, a3)

    # 计算 theta
    a1_cross_a3 = np.cross(a1, a3)
    a2_cross_a3 = np.cross(a2, a3)
    theta = np.arccos(
        np.dot(a1_cross_a3, a2_cross_a3) /
        (np.linalg.norm(a1_cross_a3) * np.linalg.norm(a2_cross_a3))
    )

    # 计算 alpha 和 beta
    alpha = rho**2 * np.linalg.norm(a1_cross_a3) * np.sin(theta)
    beta = rho**2 * np.linalg.norm(a2_cross_a3) * np.sin(theta)

    # 计算外参 r1, r2, r3, T
    r1 = a2_cross_a3 / np.linalg.norm(a2_cross_a3)
    r3 = rho * a3
    r2 = np.cross(r3, r1)
    K_inv_b = np.linalg.inv(A) @ b
    T = rho * K_inv_b

    # 打印结果
    print("内参:")
    print(f"rho: {rho}, cx: {cx}, cy: {cy}, theta: {theta}, alpha: {alpha}, beta: {beta}")
    print("\n外参:")
    print(f"r1: {r1}, r2: {r2}, r3: {r3}, T: {T}")
    
    K = np.array([alpha, -alpha/math.tan(theta), cx], [0, beta/math.sin(theta), cy], [0, 0, 1])
    R = np.stack([r1, r2, r3], axis=1)
    return K, R, T


if __name__=='__main__':
    # 加载棋盘格图片
    image = cv2.imread('./Final project/chessboard.jpg')

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    main(gray)