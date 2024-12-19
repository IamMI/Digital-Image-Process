"""
Tool Box
"""
import cv2
import os
import copy
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def get_corner_pixel(image, chessboard_size=(5,8)):
    """_summary_
    
    Args:
        image (np.ndarray): Chessboard image
        chessboard_size ((int, int)): Number of corners. Defaults to (5,8).

    Returns:
        corners (np.ndarray): Pixel coordinate of corners 
    """
    image_ = copy.deepcopy(image)
    # 查找棋盘格的角点
    ret, corners = cv2.findChessboardCorners(image_, chessboard_size, None)
    # 如果找到了角点
    if ret:
        # 精确化角点位置
        corners = cv2.cornerSubPix(image_, corners, (11, 11), (-1, -1), 
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # # 在图像上绘制角点
        # cv2.drawChessboardCorners(image_, chessboard_size, corners, ret)
        # # 显示图片
        # plt.imshow(cv2.cvtColor(image_, cv2.COLOR_BGR2RGB))
        # plt.show()

        return corners.reshape(40, 2)
    else:
        print("Chessboard corners not found.")
        exit()

def get_corner_world():
    """_summary_
    
    Returns:
        corners (np.ndarray): World coordinate of corners 
    """
    a = np.arange(0, 5)*2.6
    a = np.tile(a, 8)
    b = np.arange(0, 8)*2.6
    b = np.repeat(b, 5)
    d = np.ones((40,))
    result = np.stack((a, b, d), axis=-1)
    return result

def compute_homography(U, V, u, v):  
    # U, V, u, v 应该是形状为 (n, 1) 的 ndarray，n是点的总数  
    n = U.shape[0]  # 角点的数量  

    # 生成方程组  
    A = []  
    for i in range(n):  
        # 每一对角点生成两个方程  
        A.append([-U[i], -V[i], -1, 0, 0, 0, u[i]*U[i], u[i]*V[i], u[i]])  
        A.append([0, 0, 0, -U[i], -V[i], -1, v[i]*U[i], v[i]*V[i], v[i]])  

    A = np.array(A)  # 转换为 numpy 数组  
    # 进行奇异值分解  
    U, S, Vt = np.linalg.svd(A)  
    # 最佳 H 为 A 的最后一列的归一化版本  
    H = Vt[-1].reshape(3, 3)  

    return H  

def get_H(image):
    world = get_corner_world()
    pixel = get_corner_pixel(image)
    
    H = compute_homography(world[:, 0], world[:, 1], pixel[:, 0], pixel[:, 1])
    return H

def compute_B(H_list):  
    """  
    计算矩阵 B，基于 H 矩阵的列表，H_list 中包含多个 H 矩阵（至少3个）  
    
    参数:  
    H_list : list : 包含多个H矩阵的列表  
    
    返回:  
    B : ndarray : 9x9矩阵B  
    """  
    
    # 检查是否至少有3个H矩阵  
    if len(H_list) < 3:  
        raise ValueError("需要至少3个 H 矩阵来计算 B")  
    
    v_matrix_list = []  
    
    # 计算 v 矩阵  
    for H in H_list:  
        H1 = H[0, :]  # H 的第一行  
        H2 = H[1, :]  # H 的第二行  
        H3 = H[2, :]  # H 的第三行  
        
        i = 1
        j = 2
        v_12 = np.array([H1[i]*H1[j], H1[i]*H2[j]+H2[i]*H1[j], H2[i]*H2[j], H1[i]*H3[j]+H3[i]*H1[j], H2[i]*H3[j]+H3[i]*H2[j], H3[i]*H3[j]])
        j = 1
        v_11 = np.array([H1[i]*H1[j], H1[i]*H2[j]+H2[i]*H1[j], H2[i]*H2[j], H1[i]*H3[j]+H3[i]*H1[j], H2[i]*H3[j]+H3[i]*H2[j], H3[i]*H3[j]])
        i = 2
        j = 2
        v_22 = np.array([H1[i]*H1[j], H1[i]*H2[j]+H2[i]*H1[j], H2[i]*H2[j], H1[i]*H3[j]+H3[i]*H1[j], H2[i]*H3[j]+H3[i]*H2[j], H3[i]*H3[j]])
        
        v_matrix_list.append(v_12)
        v_matrix_list.append(v_22-v_11)
    
    A = np.stack(v_matrix_list, axis=0)

    
    # 我们需要解决方程 A * b = 0，使用 SVD  
    U, S, Vt = np.linalg.svd(A)  
    b = Vt[-1]  # 最后一行即为 b  
    
    # 构造 B 矩阵  
    B = np.zeros((3, 3))  
    B[0, 0] = b[0]
    B[0, 1] = b[1]
    B[1, 0] = b[1]
    B[1, 1] = b[2]
    B[0, 2] = b[3]
    B[2, 0] = b[3]
    B[1, 2] = b[4]
    B[2, 1] = b[4]
    B[2 ,2] = b[5]
    return B
    
def get_B(image_name_list):
    image_list = []
    for name in image_name_list:
        image = cv2.imread(name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(gray)
    
    H_list = []
    for image in image_list:
        H = get_H(image)
        H_list.append(H)
    
    B = compute_B(H_list)
    
def get_A(B):
    # 提取 B 矩阵的元素  
    B11 = B[0, 0]  
    B12 = B[0, 1]  
    B13 = B[0, 2]  
    B21 = B[1, 0]  
    B22 = B[1, 1]  
    B23 = B[1, 2]  
    B31 = B[2, 0]  
    B32 = B[2, 1]  
    B33 = B[2, 2]  

    # 计算内参  
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)  
    alpha = np.sqrt(1 / B11)  
    beta = np.sqrt(B11 / (B11 * B22 - B12**2))  
    gamma = -B12 * alpha**2 * beta  
    u0 = gamma * v0 / beta - B13 * alpha**2 

    # 生成 A 矩阵  
    A = np.array([[alpha, gamma, u0],  
                [0, beta, v0],  
                [0, 0, 1]])  

    # 输出 A 矩阵  
    print("相机内参矩阵 A:")  
    print(A)  
    return A
    
    

if __name__=='__main__':
    
    get_H(gray)


    
    
