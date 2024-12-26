import cv2
import numpy as np
import scipy
import time

import math
# import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from numba import jit
 
 
 
class Rectangle:
    """
    矩形范围(包围盒).
    """
    def __init__(self, minLon, maxLon, minLat, maxLat):
        self.minLon = minLon
        self.maxLon = maxLon
        self.minLat = minLat
        self.maxLat = maxLat
 
def myceil(d, scale):
    """
    scale 表示精度如0.1, 0.01等
    """
    n = int(1 / scale)
    return math.ceil(d * n) / n
 
def myfloor(d, scale):
    n = int(1 / scale)
    return math.floor(d * n) / n
 
def min_enclosing_rectangle(pt_list):
    """
    获取多边形区域的最小外接矩形.
    """
    rec = Rectangle(
        minLon=float('inf'), maxLon=float('-inf'),
        minLat=float('inf'), maxLat=float('-inf')
    )
    pts = np.array(pt_list)
    rec.minLon = np.min(pts[:, 0])
    rec.maxLon = np.max(pts[:, 0])
    rec.minLat = np.min(pts[:, 1])
    rec.maxLat = np.max(pts[:, 1])
    return rec
 
def compute_regional_slices(rec, scale):
    """
    根据矩形和切分尺度获取切分矩阵的点阵.
    为了减少计算量，这里输出的并不是真的矩形，由于是等距连续切分，直接输出切分的数据点即可，以左上角数据点作为标的.
    """
    # 1.根据切分尺度标准化矩形
    minLon = myfloor(rec.minLon, scale)
    maxLon = myceil(rec.maxLon, scale)
    minLat = myfloor(rec.minLat, scale)
    maxLat = myceil(rec.maxLat, scale)
    ndigit = int(math.log10(int(1 / scale))) + 1
 
    # 2.切分(出于人类习惯，暂定从上往下按行切分，即按纬度从大到小，经度从小到大)
    matrix = []
    for posLat in np.arange(maxLat, minLat, -scale):
        row = []
        for posLon in np.arange(minLon, maxLon+scale, scale):
            row.append([round(posLon, ndigit), round(posLat, ndigit)])
        matrix.append(row)
    return matrix
 
@jit(nopython=True)
def PNPoly(vertices, testp):
    """
    返回一个点是否在一个多边形区域内(开区间) PNPoly算法
    @param vertices: 多边形的顶点
    @param testp: 测试点[x, y]
    """
    n = len(vertices)
    j = n - 1
    res = False
    for i in range(n):
        if (vertices[i][1] > testp[1]) != (vertices[j][1] > testp[1]) and \
                testp[0] < (vertices[j][0] - vertices[i][0]) * (testp[1] - vertices[i][1]) / (
                vertices[j][1] - vertices[i][1]) + vertices[i][0]:
            res = not res
        j = i
    return res
 
 
def isPolygonContainsPoint(pt_list, p):
    """
    返回一个点是否在一个多边形区域内(开区间).
    """
    nCross = 0
    for i in range(len(pt_list)):
        p1 = pt_list[i]
        p2 = pt_list[(i + 1) % len(pt_list)]
        if p1[1] == p2[1]:
            continue
        if p[1] < min(p1[1], p2[1]) or p[1] >= max(p1[1], p2[1]):
            continue
        x = p1[0] + (p2[0] - p1[0]) * (p[1] - p1[1]) / (p2[1] - p1[1])
        if x > p[0]:
            nCross += 1
    return nCross % 2 == 1
 
def compute_mark_matrix(polygon, regionalSlices):
    """
    根据切分矩形和多边形计算出标记矩阵.
    """
    m = len(regionalSlices)
    n = len(regionalSlices[0])
    rectangleMarks = [[1] * (n - 1) for _ in range(m - 1)]
 
    def inRange(num, min, max):
        return num >= min and num <= max
 
    for posM in range(m):
        print(f'mark {posM}')
        for posN in range(n):
            p = regionalSlices[posM][posN]
            if not PNPoly(polygon, p):
                if inRange(posM - 1, 0, m - 2) and inRange(posN - 1, 0, n - 2):
                    rectangleMarks[posM - 1][posN - 1] = 0
                if inRange(posM - 1, 0, m - 2) and inRange(posN, 0, n - 2):
                    rectangleMarks[posM - 1][posN] = 0
                if inRange(posM, 0, m - 2) and inRange(posN - 1, 0, n - 2):
                    rectangleMarks[posM][posN - 1] = 0
                if inRange(posM, 0, m - 2) and inRange(posN, 0, n - 2):
                    rectangleMarks[posM][posN] = 0
 
    return rectangleMarks
 
def maximal_rectangle(matrix):
    """
    根据标记矩阵求最大矩形，返回【最小行标 最大行标 最小列标 最大列标 最大面积】
    """
    m = len(matrix)
    n = len(matrix[0])
    left = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                left[i][j] = (left[i][j-1] if j else 0) + 1
    min_c, max_c, min_r, max_r, ret = -1, -1, -1, -1, 0
    # 对于每一列，使用基于柱状图的方法
    for j in range(n):
        up = [0] * m
        down = [0] * m
        que = deque()
        for i in range(m):
            while len(que) > 0 and left[que[-1]][j] >= left[i][j]:
                que.pop()
            up[i] = que[-1] if que else -1
            que.append(i)
        que.clear()
        for i in range(m-1, -1, -1):
            while que and left[que[-1]][j] >= left[i][j]:
                que.pop()
            down[i] = que[-1] if que else m
            que.append(i)
        for i in range(m):
            height = down[i] - up[i] - 1
            area = height * left[i][j]
            if area > ret:
                ret = area
                min_c = up[i] + 1
                max_c = down[i] - 1
                min_r = j - left[i][j] + 1
                max_r = j
    return min_c, max_c, min_r, max_r, ret
 
def largest_internal_rectangle(polygon):
    """
    求一个多边形区域的水平方向最大内接矩形，由于是经纬度数据，精确到小数点后两位，误差(只小不大)约一公里.
    """
    scale = 0.01
    # 1.区域切块，不是真的切成矩形，而是切分成数据点
    min_enclosing_rect = min_enclosing_rectangle(polygon)
    # 2.标记矩阵，这里将点阵经纬度转换为矩形标记矩阵，每个矩形以左上角作为标的，
    #   比如矩形marks[0][0]的左上角坐标为regionalSlices[0][0]，右下角坐标为regionalSlices[1][1]
    regional_slices = compute_regional_slices(min_enclosing_rect, scale)
    marks = compute_mark_matrix(polygon, regional_slices)
    
    # 3.计算最大内接矩阵，返回矩形
    min_c, max_c, min_r, max_r, area = maximal_rectangle(marks)
    minLon = regional_slices[0][min_r][0]
    maxLon = regional_slices[0][max_r+1][0]
    minLat = regional_slices[max_c+1][0][1]
    maxLat = regional_slices[min_c][0][1]
    return Rectangle(minLon, maxLon, minLat, maxLat)
 
 
def largest_internal_rectangle_recursion(polygon, minArea=64):
    scale = 0.01
    rect_list = []
    # 1.区域切块，不是真的切成矩形，而是切分成数据点
    min_enclosing_rect = min_enclosing_rectangle(polygon)
    # 2.标记矩阵
    regional_slices = compute_regional_slices(min_enclosing_rect, scale)
    marks = compute_mark_matrix(polygon, regional_slices)
    
    # 3. 把最大矩形的mark置零，剩余部分重新计算出最大矩形，直到面积小于minArea
    while True:
        min_c, max_c, min_r, max_r, area = maximal_rectangle(marks)
        if area < minArea:
            break
        minLon = regional_slices[0][min_r][0]
        maxLon = regional_slices[0][max_r+1][0]
        minLat = regional_slices[max_c+1][0][1]
        maxLat = regional_slices[min_c][0][1]
        rect = Rectangle(minLon, maxLon, minLat, maxLat)
        for i in range(min_c, max_c+1):
            for j in range(min_r, max_r+1):
                marks[i][j] = 0
        rect_list.append(rect)
    
    return rect_list
 
def plot_rect(df, rect_list):
    # plot edge
    plt.scatter(df['lng'], df['lat'], s=5, alpha=0.8)
    
    for rect in rect_list:
        points = np.array([[rect.minLon, rect.minLat], [rect.minLon, rect.maxLat], 
                           [rect.maxLon, rect.maxLat], [rect.maxLon,rect.minLat], 
                           [rect.minLon, rect.minLat]
                        ])
        plt.plot(*zip(*points), color='r')
    plt.show()
 

def calculate_area(rectangle):
    length = abs(rectangle[0][0] - rectangle[1][0])
    width = abs(rectangle[0][1] - rectangle[2][1])
    return length * width

def is_point_inside_polygon(point, polygon):
    x, y = point
    inside = False
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        if (polygon[i][1] > y) != (polygon[j][1] > y) and \
                x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]:
            inside = not inside
    return inside

def find_max_rectangle_in_polygon(polygon):
    max_area = 0
    max_rectangle = None
    for i in range(len(polygon)):
        for j in range(i + 1, len(polygon)):
            rectangle = [polygon[i], (polygon[i][0], polygon[j][1]), polygon[j], (polygon[j][0], polygon[i][1])]
            if is_point_inside_polygon(rectangle[1], polygon) and is_point_inside_polygon(rectangle[3], polygon):
                area = calculate_area(rectangle)
                if area > max_area:
                    max_area = area
                    max_rectangle = rectangle
    return max_rectangle

# # 示例用法
# polygon = [(0, 0), (0, 5), (5, 5), (5, 0)]
# max_rectangle = find_max_rectangle_in_polygon(polygon)
# print("最大矩形的四个角点坐标：", max_rectangle)
# print("最大矩形的面积：", calculate_area(max_rectangle))

# if __name__ == '__main__':
#     df = pd.read_csv('D:/data/edge_points.csv')
#     point_list = df[['lng', 'lat']].values
 
#     ret_rect = largest_internal_rectangle_recursion(point_list, 64)
#     print('rect size {}'.format(len(ret_rect)))
    
#     plot_rect(df, ret_rect)
 

def move_edge(img, edge, edge_id):
    """
    扩展边界
    :param img: 输入图像，单通道二值图，深度为8
    :param edge: 边界数组，存放4条边界值
    :param edge_id: 当前边界号
    :return: 布尔值，确定当前边界是否可以扩展
    """
    nr, nc = img.shape[:2]
    if edge_id == 0:
        if edge[0] >= nr - 1:
            return False, edge
        for i in range(edge[3], edge[1] + 1):
            if img[edge[0] + 1, i] == 0:
                return False, edge
        edge[0] += 1
        return True, edge
    elif edge_id == 1:
        if edge[1] >= nc - 1:
            return False, edge
        for i in range(edge[2], edge[0] + 1):
            if img[i, edge[1] + 1] == 0:
                return False, edge
        edge[1] += 1
        return True, edge
    elif edge_id == 2:
        if edge[2] <= 0:
            return False, edge
        for i in range(edge[3], edge[1] + 1):
            if img[edge[2] - 1, i] == 0:
                return False, edge
        edge[2] -= 1
        return True, edge
    else:
        if edge[3] <= 0:
            return False, edge
        for i in range(edge[2], edge[0] + 1):
            if img[i, edge[3] - 1] == 0:
                return False, edge
        edge[3] -= 1
        return True, edge


def find_max_inner_rectangle(img, center, move_direction='both'):
    """
    求连通区域最大内接矩形
    :param img: 输入图像，单通道二值图
    :param center: 最小外接矩的中心
    :param move_direction: 优先移动的方向，备选参数有 "both"、"horizontal"、"vertical"
    :return: bbox，最大内接矩形
    """
    edge = [0] * 4
    edge[0] = center[1]
    edge[1] = center[0]
    edge[2] = center[1]
    edge[3] = center[0]
    is_expand = [1, 1, 1, 1]  # 扩展标记位
    # 四个方向同时外扩
    if move_direction == 'both':
        n = 0
        while any(is_expand):
            edge_id = n % 4
            is_expand[edge_id], edge = move_edge(img, edge, edge_id)
            n += 1
    # 水平方向先外扩
    elif move_direction == 'horizontal':
        n = 1
        while (is_expand[1] or is_expand[3]):
            edge_id = n % 4
            is_expand[edge_id], edge = move_edge(img, edge, edge_id)
            n += 2
        edge[3] += 20
        edge[1] -= 20
        n = 0
        while (is_expand[0] or is_expand[2]):
            edge_id = n % 4
            is_expand[edge_id], edge = move_edge(img, edge, edge_id)
            n += 2
        edge[3] -= 20
        edge[1] += 20
    # 竖直方向先外扩
    else:
        n = 0
        while (is_expand[0] or is_expand[2]):
            edge_id = n % 4
            is_expand[edge_id], edge = move_edge(img, edge, edge_id)
            n += 2
        edge[2] += 20
        edge[0] -= 20
        n = 1
        while (is_expand[1] or is_expand[3]):
            edge_id = n % 4
            is_expand[edge_id], edge = move_edge(img, edge, edge_id)
            n += 2
        edge[2] -= 20
        edge[0] += 20

    # return [edge[3], edge[2], edge[1], edge[0]]
    # return np.array([[edge[0], edge[1]],[edge[0], edge[3]], [edge[2], edge[3]],[edge[2], edge[1]]])
    return np.array([[edge[3], edge[2]],[edge[3], edge[0]], [edge[1], edge[0]],[edge[1], edge[2]]])


def get_obvious_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)

    kernel = np.ones((3,3), np.uint8)
    # dilated_mask = cv2.dilate(edges, kernel)
    # eroded_mask = cv2.erode(dilated_mask, kernel)
    # edges = eroded_mask

    edges[:150, :] = 0
    edges[-400:, :] = 0

    # edges[:, :400] = 0
    # edges[:, -400:] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # max_area = 0
    # best_rect = None

    boxs = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1][0], rect[1][1]
        wh_scale = width / (height+1e-5)
        area = width * height
        if area < 1000: continue
        if area > 200000: continue
        if wh_scale<0.2 or wh_scale>3: continue
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxs.append(box)

        cv2.drawContours(image, [box], 0, (0,0,255), 3)
    cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Region Proposal Test", image)
    cv2.imshow("Region Proposal Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxs

def get_goods_region(image, k):
    img_data = np.float32(image).reshape((-1,3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    quan_result = center[label.flatten()]
    quan_result = quan_result.reshape(image.shape)

    for idx in range(k):
        segmented_img = np.zeros_like(label,dtype=np.uint8)
        segmented_img[np.where(label==idx)] = 255
        # segmented_img = np.ones_like(label,dtype=np.uint8) * 255
        # segmented_img[np.where(label==idx)] = 0
        segmented_img = segmented_img.reshape(image.shape[:2])
        segmented_img[:150, :] = 0
        segmented_img[-400:, :] = 0

        contours, _ = cv2.findContours(segmented_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxs = []
        for cnt in contours:
            _image = image.copy()
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1][0], rect[1][1]
            wh_scale = width / (height+1e-5)
            area = width * height
            if area < 1000: continue
            if area > 20000: continue
            if wh_scale<0.2 or wh_scale>3: continue
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxs.append(box)
            
            # cv2.drawContours(image, [cnt], 0, (0,0,255), 3)
            cv2.drawContours(_image, [cnt], 0, (0,0,255), 3)
        # cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Region Proposal Edges", cv2.WINDOW_NORMAL)
        # cv2.imshow("Region Proposal Test", _image)
        # cv2.imshow("Region Proposal Edges", segmented_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # width, height = image.shape[1], image.shape[0]
    # res_w, res_h = 100, 100
    # cv2.drawContours(image, [np.array([[width//2-res_w//2, height//2-res_w//2],
    #                                    [width//2+res_w//2, height//2-res_w//2],
    #                                    [width//2+res_w//2, height//2+res_w//2],
    #                                    [width//2, height//2],
    #                                    [width//2-res_w//2, height//2+res_w//2]])], 0, (0,0,255), 3)

    cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Region Proposal Test", image)
    cv2.imshow("Region Proposal Edges", quan_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return boxs

def get_plain_region(image):
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)

    dilated_mask = cv2.dilate(edges, np.ones((5,5), np.uint8))
    eroded_mask = cv2.erode(dilated_mask, np.ones((5,5), np.uint8))
    dilated_mask = cv2.dilate(eroded_mask, np.ones((5,5), np.uint8))
    # eroded_mask = cv2.erode(dilated_mask, np.ones((7,7), np.uint8))
    plain_region = dilated_mask

    plain_region[:150, :] = 0
    plain_region[-200:, :] = 0

    # edges[:, :400] = 0
    # edges[:, -400:] = 0

    contours, _ = cv2.findContours(plain_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1][0], rect[1][1]
        wh_scale = width / (height+1e-5)
        # area = width * height
        if area < 2000 or area > 200000: continue
        if wh_scale<0.2 or wh_scale>3: continue

        gap = 30
        x,y,w,h = cv2.boundingRect(cnt)
        # gap_x, gap_y = w//row, h//column
        cx = sorted(np.arange(x, x+w, gap, dtype=np.uint32), key=lambda c:abs(c-(x+w//2)))
        cy = sorted(np.arange(y, y+h, gap, dtype=np.uint32), key=lambda c:abs(c-(y+h//2)))
        sample_centers = np.array(np.meshgrid(cx, cy)).reshape(2,-1).swapaxes(0,1)
        rects = []
        for center in sample_centers:
            if plain_region[center[1], center[0]] != 0: continue
            if cv2.pointPolygonTest(cnt, (int(center[0]), int(center[1])), False)<0: continue
            # cv2.circle(image, (center[0], center[1]), 5, (255, 0, 0), -1)
            rect = find_max_inner_rectangle(~plain_region, center, move_direction='both')
            if cv2.contourArea(rect)<1000: continue
            # cv2.drawContours(image, [rect], 0, (0,255,0), 3)
            cv2.rectangle(plain_region, rect[0], rect[2], 255, -1)
            rects.append(rect)

        merged_rect = []
        selected_idx = []
        for idx, rect in enumerate(rects):
            if idx in selected_idx: continue
            for adidx, adrect in enumerate(rects):
                if idx==adidx or adidx in selected_idx: continue
                if is_adjacent_rect(rect, adrect):
                    area = (rect[2,0]-rect[0,0]) * (rect[2,1]-rect[0,1])
                    adarea = (adrect[2,0]-adrect[0,0]) * (adrect[2,1]-adrect[0,1])
                    new_x1 = max(rect[0,0], adrect[0,0])
                    new_y1 = min(rect[0,1], adrect[0,1])
                    new_x2 = min(rect[2,0], adrect[2,0])
                    new_y2 = max(rect[2,1], adrect[2,1])
                    new_area = (new_x2-new_x1) * (new_y2-new_y1)
                    if new_area>area and new_area>adarea:
                        new_rect = np.array([[new_x1, new_y1],[new_x2, new_y1], [new_x2, new_y2],[new_x1, new_y2]])
                        merged_rect.append(new_rect)
                        rects.append(new_rect)
                        selected_idx.append(idx)
                        selected_idx.append(adidx)
                        break

                    new_x1 = min(rect[0,0], adrect[0,0])
                    new_y1 = max(rect[0,1], adrect[0,1])
                    new_x2 = max(rect[2,0], adrect[2,0])
                    new_y2 = min(rect[2,1], adrect[2,1])
                    new_area = (new_x2-new_x1) * (new_y2-new_y1)
                    if new_area>area and new_area>adarea:
                        new_rect = np.array([[new_x1, new_y1],[new_x2, new_y1], [new_x2, new_y2],[new_x1, new_y2]])
                        merged_rect.append(new_rect)
                        rects.append(new_rect)
                        selected_idx.append(idx)
                        selected_idx.append(adidx)
                        break


        for idx,rect in enumerate(rects):
            if idx not in selected_idx:
                cv2.drawContours(image, [rect], 0, (255,0,0), 3)
        
        def is_adjacent_rect(rect1, rect2, pad=5):
            x11, y11, x21, y21 = rect1[0,0], rect1[0,1], rect1[2,0], rect1[2,1]
            x12, y12, x22, y22 = rect2[0,0], rect2[0,1], rect2[2,0], rect2[2,1]
            x11 -= pad
            y11 -= pad
            x21 += pad
            y21 += pad

            x12 -= pad
            y12 -= pad
            x22 += pad
            y22 += pad
            return not (x21<=x12 or x11>=x22 or y21<=y12 or y11>=y22)
            # return ((x11<=x22+pad) and (y11<=y22+pad)) or (x21)


        # rect = find_max_rectangle_in_polygon(cnt[:,0,:])
        
        # 在图像上绘制中心点
        # cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

        # cv2.drawContours(image, [cnt], 0, (0,0,255), 3)
        # cv2.drawContours(image, [rect], 0, (0,255,0), 3)
        # image=cv2.rectangle(image.copy(),(x,y),(x+w,y+h),(0,255,0),3)#在之前的章节中，已经讲过
    print(f"cost: {time.time()-start_time:.2f}s")
    cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Region Proposal Test", image)
    cv2.imshow("Region Proposal Edges", plain_region)
    cv2.imshow("Region Proposal Mask", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxs


if __name__ == "__main__":
    # image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\WeChat_20241220211216.png",dtype=np.uint8),cv2.IMREAD_COLOR)
    image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\cut_video.mp4_20241223_133516.114.jpg",dtype=np.uint8),cv2.IMREAD_COLOR)

    # get_obvious_region(image)
    # get_goods_region(image, 5)
    get_plain_region(image)
    
    pass
