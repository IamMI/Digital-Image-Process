"""
Tool Box
"""
import cv2
import os
import copy
import numpy as np
from scipy.optimize import least_squares

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
    c = np.zeros((40,))
    d = np.ones((40,))
    result = np.stack((a, b, c, d), axis=-1)
    return result
    
def get_P(image):
    image = copy.deepcopy(image)
    pixel = get_corner_pixel(image)
    world = get_corner_world()
    
    # 初始化结果矩阵
    result = []

    # 构建矩阵
    for i in range(40):
        P_i = world[i, :].reshape(1, 4)  # 获取 P_i, 形状 (1, 4)
        u_i, v_i = pixel[i, :]           # 获取 u_i 和 v_i
        
        row_1 = np.vstack((P_i.T, np.zeros((4, 1)), -u_i * P_i.T))  
        row_2 = np.vstack((np.zeros((4,1)), P_i.T, -v_i*P_i.T)) 
        result.append(np.hstack((row_1, row_2)))  

    
    result_matrix = np.hstack(result)
    print('result_matrix.shape:', result_matrix.T.shape)  
    return result_matrix.T
    
def get_m(P):
    P_T_P = P.T @ P

    eigenvalues, eigenvectors = np.linalg.eigh(P_T_P)

    # print('Eigenvalues:',eigenvalues)
    
    # min_eigenvalue_index = np.argmin(eigenvalues)  
    # min_eigenvalue = eigenvalues[min_eigenvalue_index] 
    # min_eigenvector = eigenvectors[:, min_eigenvalue_index] 
    min_eigenvector = eigenvectors[:, 3]

    # print("最小特征值:", min_eigenvalue)
    # print("最小特征值对应的特征向量:", min_eigenvector)
    return min_eigenvector

def get_M(image):
    m = get_m(get_P(image))
    row1 = m[:4]
    row2 = m[4:8]
    row3 = m[8:]
    M = np.vstack((row1, row2, row3))
    return M    

def world_to_camera(world_points, R, t):
    if world_points.shape[-1] == 4:
        world_points = world_points[:, :-1]
        
    return np.dot(R, world_points.T).T + t

def distort_points(k1, k2, k3, points, mtx, R, t):
    # 1. Transform from world to camera
    camera_points = world_to_camera(points, R, t)
    
    # 2. Project to image plane
    x_c, y_c, z_c = camera_points.T
    x_ideal = (x_c / z_c) * mtx[0, 0] + mtx[0, 2]  # fx * (x_c / z_c) + cx
    y_ideal = (y_c / z_c) * mtx[1, 1] + mtx[1, 2]  # fy * (y_c / z_c) + cy
    
    # 3. 应用径向畸变
    r_squared = x_ideal**2 + y_ideal**2
    r = np.sqrt(r_squared)
    
    # 4. 计算畸变后的坐标
    x_distorted = x_ideal * (1 + k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3)
    y_distorted = y_ideal * (1 + k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3)
    
    return np.vstack([x_distorted, y_distorted]).T

  
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

    def cost_function(params, world, pixel, mtx):
        k1, k2, k3 = params
        
        distorted_points = distort_points(k1, k2, k3, world, mtx, R, T)
        
        error = distorted_points - pixel
        return error.ravel()
    
    
    # Initial guess
    initial_guess = [0.0, 0.0, 0.0]
    # Solve
    result = least_squares(cost_function, initial_guess, args=(world, pixel, K))
    
    k1_opt, k2_opt, k3_opt = result.x
    print(f"Optimized radial distortion coefficients: k1={k1_opt}, k2={k2_opt}, k3={k3_opt}")
    return k1_opt, k2_opt, k3_opt
    



    
    
