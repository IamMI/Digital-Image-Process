"""
Camera calibrator with trivial way
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

import utils 

def project_world_to_image(R, T, K, image):
    """
    将世界坐标点投影到图片中，并绘制红色圈圈。
    
    参数:
    R (ndarray): 相机的旋转矩阵 (3x3)。
    T (ndarray): 相机的位移向量 (3, )。
    K (ndarray): 相机的内参矩阵 (3x3)。
    image (ndarray): 输入的棋盘格图像。
    
    返回:
    image_with_projection (ndarray): 绘制了红色圆圈的图像。
    """
    world_points_homogeneous = utils.get_corner_world()
    camera_points = np.dot(R, world_points_homogeneous[:, :3].T).T + T  # (N, 3)
    
    # 将相机坐标投影到图像平面
    # 使用内参矩阵 K 进行投影，投影公式: [x_img, y_img] = K * [x_c / z_c, y_c / z_c]
    x_c, y_c, z_c = camera_points.T
    x_img = (x_c / z_c) * K[0, 0] + K[0, 2]  # fx * (x_c / z_c) + cx
    y_img = (y_c / z_c) * K[1, 1] + K[1, 2]  # fy * (y_c / z_c) + cy
    
    # 将投影结果转化为整数坐标
    image_points = np.vstack([x_img, y_img]).T
    image_points = np.round(image_points).astype(int)  # 转为整数像素坐标
    
    # 在图像上绘制红色圆圈
    image_with_projection = image.copy()
    for point in image_points:
        cv2.circle(image_with_projection, tuple(point), radius=5, color=(255, 255, 255), thickness=-1)  # 红色圈

    plt.imshow(image_with_projection, cmap='gray')
    plt.show()
    
    
    return image_with_projection


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
        -np.dot(a1_cross_a3, a2_cross_a3) /
        (np.linalg.norm(a1_cross_a3) * np.linalg.norm(a2_cross_a3))
    )

    # 计算 alpha 和 beta
    alpha = rho**2 * np.linalg.norm(a1_cross_a3) * np.sin(theta)
    beta = rho**2 * np.linalg.norm(a2_cross_a3) * np.sin(theta)

    # 计算外参 r1, r2, r3, T
    r1 = a2_cross_a3 / np.linalg.norm(a2_cross_a3)
    r3 = rho * a3
    r2 = np.cross(r3, r1)
    R = np.stack([r1, r2, r3], axis=1).T
    # K
    K = np.array([[alpha, -alpha/math.tan(theta), cx], [0, beta/math.sin(theta), cy], [0, 0, 1]])
    K_inv_b = np.linalg.inv(K)
    T = rho * K_inv_b @ b

    # 打印结果
    print("内参:")
    print(f"rho: {rho}, cx: {cx}, cy: {cy}, theta: {theta}, alpha: {alpha}, beta: {beta}")
    print("\n外参:")
    print(f"r1: {r1}, r2: {r2}, r3: {r3}, T: {T}")
    
    print("K:\n", K)
    print("R:\n", R)
    print("T:\n", T)
    # project_world_to_image(R, T, K, image)
    
    


if __name__=='__main__':
    # 加载棋盘格图片
    image = cv2.imread('./Final project/chessboard.jpg')

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    main(gray)