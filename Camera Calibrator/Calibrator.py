import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

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


def main(image_name_list):
    # Get image List
    image_list = []
    for name in image_name_list:
        image = cv2.imread(name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(gray)
        
    # Get H List
    H_list = []
    for image in image_list:
        H = utils.get_H(image)
        H_list.append(H)
        
    # Get B
    B = utils.compute_B(H_list)
    
    # Get A
    A = utils.get_A(B)
    
    # Get Extrinsics Matrix
    E_list = utils.compute_extrinsics(H_list, A)
    
    print("内参矩阵A:")
    print(A)
    print("外参矩阵E:")
    print(E_list[0])
    
    project_world_to_image(E_list[0][:3, :3], E_list[0][:3, 3], A, image_list[0])
    
    
    
if __name__ == '__main__':
    image_name_list = ['./Camera Calibrator/chessboard/chessboard1.jpg', './Camera Calibrator/chessboard/chessboard2.jpg', './Camera Calibrator/chessboard/chessboard3.jpg']
    main(image_name_list)
