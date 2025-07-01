# 数据增强技术
# 光度变换 --高斯模糊、高斯杂色、泊松杂色、亮度调整和色相调整
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import numpy as np
import torchvision.transforms as T
import shutil
import cv2
import atexit
import os

# 高斯模糊
def gaussian_blur(orig_img):
    kernel_size=(3, 3)
    sigma=2
    blurred_img=cv2.GaussianBlur(orig_img,kernel_size, sigma)
    # blurred_img = T.GaussianBlur(kernel_size=(3, 3), sigma=2)(orig_img)
    return blurred_img



# 添加高斯噪声
def gaussian_noise(img, mean=0, sigma=0.1):
    '''
    此函数用将产生的高斯噪声加到图片上
    均值为0，是保证图像的亮度不会有变化，而方差大小则决定了高斯噪声的强度。
    方差/标准差越大，噪声越强。
    
    传入参数:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回值:
        gaussian_out : 噪声处理后的图片
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out  # 这里也会返回噪声，注意返回值

# 泊松噪声
def poisson_noise(img, scale=1.0):
    '''
    为图像添加泊松噪声

    传入参数:
        img    :  原图（0-255范围）
        scale  :  泊松噪声的尺度因子，通常设为1.0，调整噪声强度

    返回值:
        noisy_img : 添加泊松噪声后的图像
    '''
    # 将图像转换为浮点型，以便进行噪声处理
    img_float = img.astype(np.float32)

    # 生成与图像尺寸相同的泊松噪声
    noise = np.random.poisson(img_float * scale) - img_float * scale

    # 将噪声添加到图像
    noisy_img = img_float + noise

    # 限制像素值在 [0, 255] 范围内
    noisy_img = np.clip(noisy_img, 0, 255)

    # 将图像转换回 uint8 类型
    noisy_img = noisy_img.astype(np.uint8)

    return noisy_img


# 导入数据
input_dir = Path('D:\\test\\testImg\\datasets\\images')  # 原始图像文件夹
mask_dir= Path('D:\\test\\testImg\\datasets\\labels')  # 标注文件夹

'''
保存文件路径
'''
output_dir = Path('D:\\test\\testImg\\blurred_images\\images')  # 输出模糊后的图像文件夹
output_dir_str='D:\\test\\testImg\\blurred_images\\images\\'
output_mask_dir= Path('D:\\test\\testImg\\blurred_images\\labels')  # 输出模糊后的图像标签文件夹
# 如果文件不存在，那么就创建文件
output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
output_mask_dir.mkdir(parents=True, exist_ok=True) 


# 依次打开input_dir中的所有图像文件
for image in os.listdir(input_dir):
    # try:
    # 打开图像文件
    img=cv2.imread(os.path.join(input_dir,image))
    mask_path=os.path.join(mask_dir,str(image[0:-4])+'.txt')
    '''
    高斯模糊
    应用高斯模糊（例如，sigma=1，kernel_size=(3, 3)
    '''
    img_blur=gaussian_blur(img)
    cv2.imwrite(os.path.join(output_dir,str(image[0:-4]) + '_gaussblur.jpg'), img_blur)
    shutil.copy(mask_path,os.path.join(output_mask_dir,str(image[0:-4]) + '_gaussblur.txt'))
    print(f"保存模糊后的图像: {image}")

    '''
    高斯噪声
    '''
    img_gaussnoise = gaussian_noise(img, mean=0, sigma=0.1)
    cv2.imwrite(os.path.join(output_dir,str(image[0:-4]) + '_gaussNoise.jpg'), img_blur)
    shutil.copy(mask_path,os.path.join(output_mask_dir,str(image[0:-4]) + '_gaussNoise.txt'))
    print(f"保存高斯噪声后的图像: {image}")

    '''
    泊松噪声
    '''
    img_poissonNoise=poisson_noise(img,scale=1.0)
    cv2.imwrite(os.path.join(output_dir,str(image[0:-4]) + '_poissonNoise.jpg'), img_blur)
    shutil.copy(mask_path,os.path.join(output_mask_dir,str(image[0:-4]) + '_poissonNoise.txt'))
    print(f"保存泊松噪声后的图像: {image}")
    
    