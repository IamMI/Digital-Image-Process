import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# 获取图片
def get_images():
    image1 = cv2.imread('./722.png')
    image2 = cv2.imread('./5947.png')
    return image1, image2
    
image1, image2 = get_images()

# 定位Gray card并绘制出来
def show_white_pixel(image):
    x = 140
    y = 100
    
    image_ = image
    image_[y-5:y+5, x-5:x+5] = [255, 0, 0]
    cv2.imshow(winname='Image', mat=image_)
    cv2.waitKey()
    cv2.destroyAllWindows()


# show_white_pixel(image1)

# Transform
def white_pixel_transform(image1, image2):
    b, g, r = image1[160, 270][0], image1[160, 270][1], image1[160, 270][2]
    
    image_ = image2.copy().astype(np.float32)
    image_[:,:,0] = image_[:,:,0]*255/b
    image_[:,:,1] = image_[:,:,1]*255/g
    image_[:,:,2] = image_[:,:,2]*255/r
    
    image_ = np.clip(image_, 0, 255).astype(np.uint8)
    
    # 将前后两张图片进行对比
    image = np.concatenate([image_, image2], axis=1)
    
    cv2.imshow(winname='Image after tranform(left), Original Image(right)', mat=image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def gray_world_transform(image1, image2):
    b, g, r = image2[:,:,0].mean(), image2[:,:,1].mean(), image2[:,:,2].mean()
    avg = (b+r+g)/3
    image_ = image2.copy().astype(np.float32)
    image_[:,:,0] = image_[:,:,0]*avg/b
    image_[:,:,1] = image_[:,:,1]*avg/g
    image_[:,:,2] = image_[:,:,2]*avg/r
    
    image_ = np.clip(image_, 0, 255).astype(np.uint8)
    
    # 将前后两张图片进行对比
    image = np.concatenate([image_, image2], axis=1)
    
    cv2.imshow(winname='Image after tranform(left), Original Image(right)', mat=image)
    cv2.waitKey()
    cv2.destroyAllWindows()

gray_world_transform(image1, image2)


def brightest_pixel_transform(image1, image2):
    image_ = image2.copy().astype(np.float32)
    
    
    def calculate_brightness(img):
        return 0.2989 * img[:, :, 2] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 0]

    # 找到最亮的像素位置
    brightness = calculate_brightness(image_)
    brightest_pixel = np.unravel_index(np.argmax(brightness), brightness.shape)
    value = image_[brightest_pixel]
    
    # 计算缩放因子
    scaling_factor = 255 / value

    # 调整图片，使最亮像素的亮度变为255
    image_ = np.clip(image_ * scaling_factor, 0, 255).astype(np.uint8)

    image = np.concatenate([image_, image2], axis=1)
    # 显示调整后的图像
    cv2.imshow('Image after tranform(left), Original Image(right)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# brightest_pixel_transform(image1, image2)