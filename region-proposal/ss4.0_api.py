import cv2
import numpy as np
import os
from datetime import datetime
'''
读取图片文件夹
'''
def get_jpg_files(folder_path):
    # 创建一个空列表，用于存储所有以.jpg结尾的文件名
    jpg_files = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以.jpg结尾
        if filename.lower().endswith('.jpg'):
            jpg_files.append(filename)  # 将符合条件的文件名添加到列表中
    
    return jpg_files

def alter_contrast(image, alpha=1.5, beta=0):
    # 调整对比度
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_image
'''
step2：降低亮度，消除水面反光
'''
def remove_highlights(image, maxthreshold=100):
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取亮度通道（V通道）
    h, s, v = cv2.split(hsv)
    
    # 限制亮度通道的值，设定阈值
    v[v > maxthreshold] = maxthreshold  # 将亮度大于threshold的部分设置为threshold
    
    # 合并处理后的通道
    hsv = cv2.merge([h, s, v])
    
    # 转换回BGR颜色空间
    result_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result_image

'''
显示图像
'''
def show_image(image_name, image):
    # 显示结果
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

'''
合并多个图像
'''
def merge_image(image_list):
    # 水平拼接两张图片
    top_row = cv2.hconcat([image_list[0], image_list[1]])
    #show_image("top_row", top_row)
    print(top_row.shape)
    # 水平拼接另外两张图片
    bottom_row = cv2.hconcat([image_list[2], image_list[3]])
    bottom_row = cv2.cvtColor(bottom_row, cv2.COLOR_GRAY2BGR)
    #show_image("bottom_row", bottom_row)
    print(bottom_row.shape)
    # 垂直拼接两行
    final_image = cv2.vconcat([top_row, bottom_row])
    final_image = cv2.resize(final_image, (int(final_image.shape[1]/2), int(final_image.shape[0]/2)))
    return final_image


'''
主功能：框选目标区域
'''
def detect(image):
    # 加载图像
    height, width, channel = image.shape
    width = int(width/2)
    height = int(height/2)
    image = cv2.resize(image, (width, height))
    ori_image = image

    image = alter_contrast(image, alpha=1.5, beta=0)
    contrast_image = image
    image = remove_highlights(image, maxthreshold=100)
    darker_image = image

    #show_image("darker", darker_image)

    # 转换图像到灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建掩模以去除纯白色和纯黑色区域
    mask = cv2.inRange(gray_image, 1, 254)

    # 应用掩模到原始图像
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    #show_image("filtered_image(delete white&black)", filtered_image)

    # 应用Canny边缘检测
    edges = cv2.Canny(filtered_image, 100, 300, apertureSize=3)

    #show_image("edges", edges)

    # 形态学操作：闭运算，填充边缘内部
    kernel = np.ones((5,5), np.uint8)
    filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    #show_image("filled", filled)
    merge = merge_image([contrast_image, darker_image, edges, filled])
    # show_image("merge", merge)
    # cv2.imwrite(processed_folder_path + "/" + filename, merge)
    cv2.imshow("Region Proposal Test", merge)
    
    
    
    # 查找轮廓
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 可以在这里添加条件，比如根据面积过滤轮廓
        if w * h > 400 and w*h <10000:  # 假设甲板区域的面积大于1000像素
            # 绘制边界框
            if w >= h:
                w = min(w, int(image.shape[1]/10))
                h = w
                cv2.rectangle(ori_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                h = min(h, int(image.shape[1]/10))
                w = h
                cv2.rectangle(ori_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imwrite(result_folder_path + "/" + filename, ori_image)
    cv2.imshow("Region Proposal Test2", ori_image)
    # 显示结果
    # cv2.imshow('Detected Deck Areas', ori_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = cv2.imdecode(np.fromfile(r"C:\Users\Namiko\Downloads\测试图像\临时图像\微信图片_20241220211216.png",dtype=np.uint8),cv2.IMREAD_COLOR)
    start_time = datetime.now()
    detect(image)
    end_time = datetime.now()
    print("用时", end_time - start_time, "秒")