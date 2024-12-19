import cv2
import numpy as np
import glob

# 棋盘格参数
chessboard_size = (8, 5)  # 棋盘格内角点数（8列，5行）
square_size = 1.0  # 方格边长（单位：例如 1mm，可以根据实际单位调整）

# 世界坐标系中的棋盘格角点坐标（Z=0）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 存储棋盘格角点的 3D 世界坐标和 2D 图像坐标
objpoints = []  # 3D 世界坐标
imgpoints = []  # 2D 图像坐标

# 加载棋盘格图片
images = glob.glob('./Camera Calibrator/chessboard/*.jpg')  # 假设棋盘格图片存储在 chessboard_images 文件夹下

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        objpoints.append(objp)  # 添加 3D 世界坐标
        imgpoints.append(corners)  # 添加 2D 图像坐标

        # 可视化角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 相机标定
if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 打印标定结果
    print("相机内参矩阵 (camera_matrix):\n", camera_matrix)
    print("\n畸变系数 (dist_coeffs):\n", dist_coeffs.ravel())

    # 打印每张图片对应的外参矩阵
    for i in range(len(rvecs)):
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvecs[i])
        
        # 获取外参矩阵 [R|t]
        t = tvecs[i].reshape(3, 1)  # 将平移向量调整为列向量
        extrinsic_matrix = np.hstack((R, t))
        if i == 1:
            print(f"\n图片 {i+1} 的外参矩阵 (Extrinsic Matrix):\n", extrinsic_matrix)
else:
    print("未找到足够的棋盘格角点进行标定！")
