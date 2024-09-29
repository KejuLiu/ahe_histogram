import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def load_image(image_path):
    # 加载图像，转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return None
    return image

def apply_ahe(image):
    # 应用自适应直方图均衡化
    return exposure.equalize_adapthist(image, clip_limit=0.03)

def plot_histograms(original, enhanced):
    # 绘制原始图像和增强图像的直方图
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), bins=256, color='gray', alpha=0.7, label='Original Histogram')
    plt.xlim([0, 256])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist((enhanced*255).ravel(), bins=256, color='blue', alpha=0.7, label='AHE Enhanced Histogram')
    plt.xlim([0, 256])
    plt.legend()

    plt.show()

def main():
    # 定义图像路径
    image_path = r'D:\lab4scripts\YOLOv8-ultralytics-2024.10.26-main\datasets\train\images\DJI_20240719063452_0012_T_JPG.rf.d8bd79c0972e98bb7d10e4f682a88fbf.jpg'

    # 加载图像
    original_image = load_image(image_path)
    if original_image is None:
        return

    # 应用 AHE
    ahe_image = apply_ahe(original_image)

    # 绘制直方图进行对比
    plot_histograms(original_image, ahe_image)

if __name__ == '__main__':
    main()

