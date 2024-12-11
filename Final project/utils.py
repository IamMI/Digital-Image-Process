"""
Tool Box
"""
import cv2
import os
import copy
import numpy as np

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

    # 将结果拼接成最终矩阵 (80, 6)
    result_matrix = np.hstack(result)
    print('result_matrix.shape:', result_matrix.T.shape)  # 输出 (80, 6)
    return result_matrix.T
    
def get_m(P=np.random.randn(80, 12)):
    P_T_P = P.T @ P

    eigenvalues, eigenvectors = np.linalg.eigh(P_T_P)

    # min_eigenvalue_index = np.argmin(eigenvalues)  
    # min_eigenvalue = eigenvalues[min_eigenvalue_index] 
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
    
    
def main(image):
    M = get_M(image)
    
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
    
    
        
    

if __name__=='__main__':
    # 加载棋盘格图片
    image = cv2.imread('./Final project/chessboard.jpg')

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    main(gray)