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
        
        i = 0
        j = 1
        v_12 = np.array([H1[i]*H1[j], H1[i]*H2[j]+H2[i]*H1[j], H2[i]*H2[j], H1[i]*H3[j]+H3[i]*H1[j], H2[i]*H3[j]+H3[i]*H2[j], H3[i]*H3[j]])
        i = 0
        j = 0
        v_11 = np.array([H1[i]*H1[j], H1[i]*H2[j]+H2[i]*H1[j], H2[i]*H2[j], H1[i]*H3[j]+H3[i]*H1[j], H2[i]*H3[j]+H3[i]*H2[j], H3[i]*H3[j]])
        i = 1
        j = 1
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
    u0 = (-B12 * v0 - B13)  / B11
    rho = B33 - B22 * v0**2 - B11 * u0**2 - 2 * B12 * u0 * v0
    alpha = np.sqrt(rho / B11)
    beta = np.sqrt(rho * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * beta / B11
    # alpha = np.sqrt(1 / B11)  
    # beta = np.sqrt(B11 / (B11 * B22 - B12**2))  
    # gamma = -B12 * alpha**2 * beta  
    # u0 = gamma * v0 / beta - B13 * alpha**2 

    # 生成 A 矩阵  
    A = np.array([[alpha, gamma, u0],  
                [0, beta, v0],  
                [0, 0, 1]])  
    return A
    
def compute_extrinsics(H_list, A):  
    # 计算内参矩阵的逆，即 A^{-1}  
    A_inv = np.linalg.inv(A)  
    
    E_list = []
    for H in H_list:
        extrinsic_matrix = A_inv @ H  # 使用 @ 符号进行矩阵乘法  
        # 提取 R 和 T  
        rho = np.linalg.norm(extrinsic_matrix[:, 0])
        R1 = extrinsic_matrix[:, 0]/rho # 第一列  
        R2 = extrinsic_matrix[:, 1]/rho # 第二列  
        T = extrinsic_matrix[:, 2]/rho    # 第三列，平移向量  
        
        # 计算 R3，通过 R1 和 R2 的叉乘  
        R3 = np.cross(R1, R2)  
        
        # 合成完整的外参矩阵 (R | T)  
        R = np.column_stack((R1, R2, R3))  # 将 R1, R2, R3 组成旋转矩阵  
        extrinsic_matrix_complete = np.column_stack((R, T))  
        extrinsic_matrix_complete = np.row_stack((extrinsic_matrix_complete, [0, 0, 0, 1]))  # 添加最后一行 [0, 0, 0, 1]  
        E_list.append(extrinsic_matrix_complete)
    
    return E_list

def world_to_camera(world_points, R, t):
    if world_points.shape[-1] == 4:
        world_points = world_points[:, :-1]
        
    return np.dot(R, world_points.T).T + t

def distort_points(camera_points, pixel, mtx, image):
    """_summary_
    
    Args:
        k1 (float): Radial distortion coefficient
        k2 (float): Radial distortion coefficient
        camera_points (np.ndarray): Camera coordinate of corners
        pixel (np.ndarray): Pixel coordinate of corners
        mtx (np.ndarray): Intrinsic parameters
        R (np.ndarray): Rotation Matrix
        T (np.ndarray): Translation Matrix
    
    Returns:
        error (np.ndarray): Error between distorted and undistorted points
    """
    # 将相机坐标投影到图像平面
    # 使用内参矩阵 K 进行投影，投影公式: [x_img, y_img] = K * [x_c / z_c, y_c / z_c]
    x_c, y_c, z_c = camera_points.T
    U = (x_c / z_c) * mtx[0, 0] + (y_c / z_c) * mtx[0, 1] + mtx[0, 2]  # fx * (x_c / z_c) + cx
    V = (y_c / z_c) * mtx[1, 1] + mtx[1, 2]  # fy * (y_c / z_c) + cy
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    r = np.sqrt(x_c**2 + y_c**2)
    
    n = camera_points.shape[0]
    U_hat = pixel[:, 0]
    V_hat = pixel[:, 1]

    # 构造矩阵 D
    D = np.zeros((2 * n, 2))
    for i in range(n):
        D[2 * i] = [(U[i] - cx) * r[i]**2/z_c[i]**2, (U[i] - cx) * r[i]**4/z_c[i]**4]
        D[2 * i + 1] = [(V[i] - cy) * r[i]**2/z_c[i]**2, (V[i] - cy) * r[i]**4/z_c[i]**4]

    # 构造向量 d
    d = np.zeros(2 * n)
    for i in range(n):
        d[2 * i] = U_hat[i] - U[i]
        d[2 * i + 1] = V_hat[i] - V[i]

    # 最小二乘解
    k, _, _, _ = np.linalg.lstsq(D, d, rcond=None)
        
    # 绘图
    if True:
        x_img = cx + (U - cx)*(1+k[0]*r**2/z_c**2 + k[1]*r**4/z_c**4)
        y_img = cy + (V - cy)*(1+k[0]*r**2/z_c**2 + k[1]*r**4/z_c**4)
        
        # 将投影结果转化为整数坐标
        image_points = np.vstack([x_img, y_img]).T
        image_points = np.round(image_points).astype(int)  # 转为整数像素坐标
        
        # 在图像上绘制红色圆圈
        image_with_projection = image.copy()
        for point in image_points:
            cv2.circle(image_with_projection, tuple(point), radius=5, color=(255, 255, 255), thickness=-1)  # 红色圈

        # plt.imshow(image_with_projection, cmap='gray')
        # plt.show()
        plt.imsave('./Camera Calibrator/chessboard/disorted_image_with_projection.png', image_with_projection, cmap='gray')
        
        
    return k
    
    
    
    

  
def get_distorted(K, R, T, image):
    """ Using optimization method to obtain distorted coefficient

    Args:
        K (np.ndarray): Intrinsic parameters
        R (np.ndarray): Rotation Matrix
        T (np.ndarray): Translation Matrix
        image (np.ndarray): Image
        
    Return: Distorted coefficients
    """
    assert type(K)==np.ndarray and type(R)==np.ndarray and type(T)==np.ndarray
    
    image = copy.deepcopy(image)
    pixel = get_corner_pixel(image)
    world = get_corner_world()
    camera_points = world_to_camera(world, R, T)


    # Solve
    k = distort_points(camera_points, pixel, K, image)
    
    k1_opt, k2_opt = k[0], k[1] 
    np.set_printoptions(precision=2)
    print(f"Optimized radial distortion coefficients: k1={k1_opt}, k2={k2_opt}")
    
    return k1_opt, k2_opt


    




    
    
