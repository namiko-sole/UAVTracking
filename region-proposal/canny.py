import os
from datetime import datetime
from fileinput import filename

import cv2
import numpy as np
from sklearn.cluster import KMeans
import numpy as np

from max_inner_rect import find_max_inner_rectangle
import matplotlib.pyplot as plt



input_path = r"C:\Users\gaoxi\69b\PID\pythonProject\testdata"
output_path = r"C:\Users\gaoxi\69b\PID\pythonProject\output_masks"
output_path_final= r"C:\Users\gaoxi\69b\PID\pythonProject\output_final"
output_path_clusters = r"C:\Users\gaoxi\69b\PID\pythonProject\output_clusters_masks"
edge_output_path = r"C:\Users\gaoxi\69b\PID\pythonProject\output_edges"
cluster_output_path = r"C:\Users\gaoxi\69b\PID\pythonProject\output_clusters"
eroded_mask_path = r"C:\Users\gaoxi\69b\PID\pythonProject\eroded_mask"

# 创建输出目录（如果不存在）
os.makedirs(output_path, exist_ok=True)
os.makedirs(edge_output_path, exist_ok=True)
os.makedirs(cluster_output_path, exist_ok=True)
os.makedirs(output_path_clusters, exist_ok=True)
os.makedirs(output_path_final, exist_ok=True)
os.makedirs(eroded_mask_path, exist_ok=True)
# 设置Canny算子的阈值
canny_threshold1 = 50
canny_threshold2 = 100
k=3/5
num=5


def get_max_inner_rectangles(matrix_np: np.ndarray, rectangle_bbox: list, area_value: int, result_list: list,
                             cur_area: float = float('inf')) -> list:
    """
    递归获取空间的多个内接矩形
    Args:
        matrix_np: 包含空间的底图
        rectangle_bbox: 空间的外接矩形
        area_value: 最小面积阈值
        result_list: 内接矩形列表
        cur_area: 当前矩形的面积
    Returns:
        result_list: 内接矩形列表
    """
    xmin, ymin, xmax, ymax = rectangle_bbox
    crop_img = matrix_np[ymin:ymax, xmin:xmax]  # 通过最大外接矩形，crop包含该空间的区域，优化速度
    matrix_list = crop_img.tolist()

    row = len(matrix_list)
    col = len(matrix_list[0])
    height = [0] * (col + 2)
    res = 0  # 记录矩形内像素值相加后的最大值
    bbox_rec = None  # 最大内接矩形bbox
    for i in range(row):
        stack = []  # 利用栈的特性获取最大矩形区域
        for j in range(col + 2):
            if 1 <= j <= col:
                if matrix_list[i][j - 1] == 255:
                    height[j] += 1
                else:
                    height[j] = 0
            # 精髓代码块 计算最大内接矩形 并计算最大值
            while stack and height[stack[-1]] > height[j]:
                cur = stack.pop()
                if res < (j - stack[-1] - 1) * height[cur]:
                    res = (j - stack[-1] - 1) * height[cur]
                    bbox_rec = [stack[-1], i - height[cur], j, i]
            stack.append(j)

    # 递归停止条件，1.最大内接矩形面积小于阈值；2. 没有最大内接矩形
    if cur_area < area_value or not bbox_rec:
        return result_list
    # 映射到原图中的位置
    src_min_x = xmin + bbox_rec[0]
    src_min_y = ymin + bbox_rec[1]
    src_max_x = xmin + bbox_rec[2]
    src_max_y = ymin + bbox_rec[3]
    bbox_src_position = [src_min_x, src_min_y, src_max_x, src_max_y]
    # 转成np格式，并将已经找到的最大内接矩形涂黑
    bbox_cnt = [[bbox_src_position[0], bbox_src_position[1]],
                [bbox_src_position[2], bbox_src_position[1]],
                [bbox_src_position[2], bbox_src_position[3]],
                [bbox_src_position[0], bbox_src_position[3]]]
    contour_cur_np = np.array(bbox_cnt).reshape(-1, 1, 2)
    cv2.polylines(matrix_np, [contour_cur_np], 1, 0)
    cv2.fillPoly(matrix_np, [contour_cur_np], 0)
    cur_area =  (bbox_rec[2] - bbox_rec[0]) * (bbox_rec[3] - bbox_rec[1])
    if cur_area > area_value:
        result_list.append(bbox_src_position)
    # 递归获取剩下的内接矩形
    get_max_inner_rectangles(matrix_np, rectangle_bbox, area_value, result_list, cur_area)

    return result_list

def findrect(img):
    x, y, w, h = cv2.boundingRect(img)
    cnt_bbox = [x, y, x + w, y + h]
    print(cnt_bbox)
    res_list = get_max_inner_rectangles(img, cnt_bbox, 100, [])
    res_list = sorted(res_list, key=lambda _: (_[2] -_[0]) *(_[3] -_[1]), reverse=True)
    return res_list

def findWater(list):
    dict={"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0}
    for i in range(len(list)):
        dict[str(list[i])]+=1
    return max(dict, key=dict.get)

# 输入路径和输出路径

def process_image(img):
    # Step 1: Calculate the number of 255 values per column
    col_sum = np.sum(img == 255, axis=0)

    # Find max, min, and avg of the 255 counts in the columns
    col_max = np.max(col_sum)
    col_min = np.min(col_sum)
    col_avg = col_max*k

    # Select columns with 255 counts greater than avg
    indexList = np.where(col_sum > col_avg)[0]

    # Find the smallest column index with at least 5 consecutive values in indexList
    indexmin = None
    for i in range(len(indexList) - 4):
        if indexList[i] + 1 == indexList[i + 1] and indexList[i + 1] + 1 == indexList[i + 2] \
           and indexList[i + 2] + 1 == indexList[i + 3] and indexList[i + 3] + 1 == indexList[i + 4]:
            indexmin = indexList[i]
            break

    # Find the largest column index with at least 5 consecutive values in reverse order
    indexmax = None
    for i in range(len(indexList) - 1, 3, -1):
        if indexList[i] - 1 == indexList[i - 1] and indexList[i - 1] - 1 == indexList[i - 2] \
           and indexList[i - 2] - 1 == indexList[i - 3] and indexList[i - 3] - 1 == indexList[i - 4]:
            indexmax = indexList[i]
            break

    # Step 2: Repeat the same process for rows
    row_sum = np.sum(img == 255, axis=1)

    row_max = np.max(row_sum)
    row_min = np.min(row_sum)
    row_avg = row_max*k

    indexList_row = np.where(row_sum > row_avg)[0]

    rowmin = None
    for i in range(len(indexList_row) - 4):
        if indexList_row[i] + 1 == indexList_row[i + 1] and indexList_row[i + 1] + 1 == indexList_row[i + 2] \
           and indexList_row[i + 2] + 1 == indexList_row[i + 3] and indexList_row[i + 3] + 1 == indexList_row[i + 4]:
            rowmin = indexList_row[i]
            break

    rowmax = None
    for i in range(len(indexList_row) - 1, 3, -1):
        if indexList_row[i] - 1 == indexList_row[i - 1] and indexList_row[i - 1] - 1 == indexList_row[i - 2] \
           and indexList_row[i - 2] - 1 == indexList_row[i - 3] and indexList_row[i - 3] - 1 == indexList_row[i - 4]:
            rowmax = indexList_row[i]
            break

    # Step 3: Define the rectangle and divide it into 8x8 grids
    if indexmin is None or indexmax is None or rowmin is None or rowmax is None:
        raise ValueError("Unable to find a valid rectangle with consecutive indices.")

    rectangle = img[rowmin:rowmax+1, indexmin:indexmax+1]
    height, width = rectangle.shape
    grid_h, grid_w = height // num-1, width // num-1

    superPointsList = []
    for i in range(num):
        for j in range(num):
            x = indexmin + j * grid_w if j < num else indexmax
            y = rowmin + i * grid_h if i < num else rowmax
            superPointsList.append((y, x))

    return superPointsList

def visualize_superpoints(img, superPointsList):
    plt.imshow(img, cmap='gray')
    for point in superPointsList:
        plt.scatter(point[1], point[0], c='red', s=10)
    plt.title('Superpoints Visualization')
    plt.show()


def filter_and_find_rectangles(img, binary_water_edge_mask):
    # Filter points based on img mask
    superPointsList = process_image(img)

    filtered_points = [(y, x) for y, x in superPointsList if binary_water_edge_mask[y, x] == 255]
    visualize_superpoints(binary_water_edge_mask, filtered_points)
    rectangles = []
    # rect = find_max_inner_rectangle(binary_water_edge_mask, (800,500))
    # rectangles.append(rect)
    for central in filtered_points:
        print(central)
        rect = find_max_inner_rectangle(binary_water_edge_mask, (central[1], central[0]))
        rectangles.append(rect)

    # Remove overlapping rectangles
    non_overlapping_rectangles = rectangles
    for i, rect1 in enumerate(rectangles):
        for j, rect2 in enumerate(rectangles):
            if i != j:
                overlap_area = calculate_overlap_area(rect1, rect2)
                if overlap_area > 0.5 * rectangle_area(rect1):
                    rectangles.remove(rect2)

    return rectangles

def calculate_overlap_area(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0

def rectangle_area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def cell(input_path, filename):
    time1=datetime.now()
    img_path = os.path.join(input_path, filename)
    img = cv2.imread(img_path)
    if img is None:
        return

    # 转换为灰度图进行边缘检测
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)

    # 保存边缘检测结果
    edge_filename = f"{os.path.splitext(filename)[0]}_edges.png"
    edge_path = os.path.join(edge_output_path, edge_filename)
    cv2.imwrite(edge_path, edges)

    # 使用KMeans进行颜色聚类
    reshaped_img = img.reshape((-1, 3))
    reshaped_img[0,] = 0, 0, 0
    reshaped_img[1,] = 130, 130, 130
    kmeans = KMeans(n_clusters=5, random_state=42).fit(reshaped_img)
    clustered_img = kmeans.labels_.reshape((img.shape[0], img.shape[1]))
    black = clustered_img[0, 0]
    water = clustered_img[0, 1]

    # 确定水面区域（左侧和右侧的主要类别）
    # i=0
    # water=0
    # while True:
    #     water = findWater(clustered_img[:,i])
    #     if water != black:
    #         break

    # 创建二分类水面掩码图（水面为白，其他为黑）
    binary_water_mask = np.full_like(clustered_img, 0, dtype=np.uint8)
    binary_water_mask[(clustered_img == water) | (clustered_img == black)] = 255

    # 保存聚类结果
    cluster_filename = f"{os.path.splitext(filename)[0]}_clusters.png"
    cluster_path = os.path.join(cluster_output_path, cluster_filename)
    cv2.imwrite(cluster_path, (clustered_img * (255 // kmeans.n_clusters)).astype(np.uint8))

    # 保存二分类水面掩码
    binary_mask_filename = f"{os.path.splitext(filename)[0]}_binary_water_mask.png"
    binary_mask_path = os.path.join(output_path_clusters, binary_mask_filename)
    cv2.imwrite(binary_mask_path, binary_water_mask)

    binary_water_edge_mask = np.full_like(clustered_img, 0, dtype=np.uint8)
    binary_water_edge_mask[(binary_water_mask == 0) & (edges == 0)] = 255
    binary_mask_filename = f"{os.path.splitext(filename)[0]}_binary_edge_water_mask.png"
    binary_mask_path = os.path.join(output_path, binary_mask_filename)
    cv2.imwrite(binary_mask_path, binary_water_edge_mask)
    print(f"已处理: {filename}, 聚类结果和二分类水面掩码已保存")

    cluster_filename = f"{os.path.splitext(filename)[0]}_clusters.png"
    cluster_path = os.path.join(eroded_mask_path, cluster_filename)
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(binary_water_edge_mask, kernel)
    eroded_mask = cv2.erode(dilated_mask, kernel)
    cv2.imwrite(cluster_path, eroded_mask)
    rects=filter_and_find_rectangles(eroded_mask, binary_water_edge_mask)
    for rect in rects:
        box = [np.array([[rect[0], rect[1]],[rect[0], rect[3]],
                         [rect[2], rect[3]],[rect[2], rect[1]]])]
        cv2.drawContours(img, box, 0, (0, 0, 255), 2)

    binary_mask_filename = f"{os.path.splitext(filename)[0]}_binary_water_mask.png"
    binary_mask_path = os.path.join(output_path_final, binary_mask_filename)
    cv2.imwrite(binary_mask_path, img)
    print(binary_mask_path)
    time2=datetime.now()
    print((time2-time1).total_seconds())

if __name__ == '__main__':
    #遍历输入路径中的所有图片
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            cell(input_path, filename)# 读取图片
    print("所有图片处理完成！")
    # cell(input_path, "Screenshot_20241210-152016.png")
