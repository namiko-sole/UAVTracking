import cv2
import numpy as np
import scipy
import time

import math
# import pandas as pd
import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt
# from numba import jit

def is_adjacent_rect(rect1, rect2, pad=5):
    """
    判断是否是相邻矩形
    :param rect1: 矩形A
    :param rect1: 矩形B
    :param pad: 可接受邻域
    :return: 布尔值, 确定两个矩形是否相邻 
    """
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
        # if area < 1000: continue
        # if area > 200000: continue
        # if wh_scale<0.2 or wh_scale>3: continue
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

def get_plain_region(image, edge_thres1=50, edge_thres2=100, 
                     min_area_thres=2000, max_area_thres=500000,
                     sample_gap=30, adjacent_rect_gap=5,
                     dilate_kernel=5, debug=False, debug_details=False):
    """
    获取图像内的空地投放区域
    :param img: 输入图像，单通道二值图
    :param center: 最小外接矩的中心
    :param move_direction: 优先移动的方向，备选参数有 "both"、"horizontal"、"vertical"
    :return: bbox，最大内接矩形
    """
    # 根据边缘提取区域轮廓
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thres1, edge_thres2)
    plain_region = cv2.dilate(edges, np.ones((dilate_kernel,dilate_kernel), np.uint8))
    plain_region[:150, :] = 0
    plain_region[-400:, :] = 0
    contours, _ = cv2.findContours(plain_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        image = image.copy()
        _plain_region = plain_region.copy()

    all_rects = []
    for cnt in contours:
        # 过滤太小、太大或过于细长的区域
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1][0], rect[1][1]
        wh_scale = width / (height+1e-5)
        if wh_scale<0.2 or wh_scale>3: continue
        if area < min_area_thres or area > max_area_thres: continue

        # 采样矩形生长点
        x,y,w,h = cv2.boundingRect(cnt)
        cx = sorted(np.arange(x, x+w, sample_gap, dtype=np.uint32), key=lambda c:abs(c-(x+w//2)))
        cy = sorted(np.arange(y, y+h, sample_gap, dtype=np.uint32), key=lambda c:abs(c-(y+h//2)))
        sample_centers = np.array(np.meshgrid(cx, cy)).reshape(2,-1).swapaxes(0,1)

        # 根据采样点生长矩形
        rects = []
        for center in sample_centers:
            if plain_region[center[1], center[0]] != 0: continue  # 如果采样点不在空白区域内，跳过
            if cv2.pointPolygonTest(cnt, (int(center[0]), int(center[1])), False)<0: continue  # 如果采样点不在轮廓内，跳过
            rect = find_max_inner_rectangle(~plain_region, center, move_direction='both')
            if cv2.contourArea(rect)<1000: continue  # 如果得到的矩形面积太小，跳过
            cv2.rectangle(plain_region, rect[0], rect[2], 255, -1)  # 对已生成矩形的区域标记
            rects.append(rect)
            if debug and debug_details:
                cv2.circle(image, (center[0], center[1]), 5, (255, 0, 0), -1)
                cv2.drawContours(image, [rect], 0, (0,255,0), 3)

        # 多个小矩阵合成最大内接矩形
        merged_idx = []
        for idx, rect in enumerate(rects):
            if idx in merged_idx: continue
            for adidx, adrect in enumerate(rects):
                if idx==adidx or adidx in merged_idx: continue
                if is_adjacent_rect(rect, adrect, adjacent_rect_gap):
                    # 合成情况A
                    area = (rect[2,0]-rect[0,0]) * (rect[2,1]-rect[0,1])
                    adarea = (adrect[2,0]-adrect[0,0]) * (adrect[2,1]-adrect[0,1])
                    new_x1 = max(rect[0,0], adrect[0,0])
                    new_y1 = min(rect[0,1], adrect[0,1])
                    new_x2 = min(rect[2,0], adrect[2,0])
                    new_y2 = max(rect[2,1], adrect[2,1])
                    new_area = (new_x2-new_x1) * (new_y2-new_y1)
                    if new_area>area and new_area>adarea:
                        new_rect = np.array([[new_x1, new_y1],[new_x2, new_y1], [new_x2, new_y2],[new_x1, new_y2]])
                        rects.append(new_rect)
                        merged_idx.append(idx)
                        merged_idx.append(adidx)
                        break

                    # 合成情况B
                    new_x1 = min(rect[0,0], adrect[0,0])
                    new_y1 = max(rect[0,1], adrect[0,1])
                    new_x2 = max(rect[2,0], adrect[2,0])
                    new_y2 = min(rect[2,1], adrect[2,1])
                    new_area = (new_x2-new_x1) * (new_y2-new_y1)
                    if new_area>area and new_area>adarea:
                        new_rect = np.array([[new_x1, new_y1],[new_x2, new_y1], [new_x2, new_y2],[new_x1, new_y2]])
                        rects.append(new_rect)
                        merged_idx.append(idx)
                        merged_idx.append(adidx)
                        break

        # 保存未被合成的矩形
        final_rects = []
        for idx, rect in enumerate(rects):
            if idx not in merged_idx:
                final_rects.append(rect)
                if debug: cv2.drawContours(image, [rect], 0, (255,0,0), 3)
        if debug and debug_details: cv2.drawContours(image, [cnt], 0, (0,0,255), 3)
        all_rects.extend(rects)

    if debug:
        return all_rects, image, _plain_region
    else:
        return all_rects


if __name__ == "__main__":
    image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\WeChat_20241220211216.png",dtype=np.uint8),cv2.IMREAD_COLOR)
    # image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\cut_video.mp4_20241223_133516.114.jpg",dtype=np.uint8),cv2.IMREAD_COLOR)
    # image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\screen-20241202-121915.mp4_20241225_163214.686.jpg",dtype=np.uint8),cv2.IMREAD_COLOR)

    
    start_time = time.time()

    # rects = get_plain_region(image)
    rects, debug_image, debug_region = get_plain_region(image, debug=True)#, debug_details=True)

    print(f"cost: {time.time()-start_time:.2f}s")

    ### 可视化调试 ###
    cv2.namedWindow("Debug Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug Region", cv2.WINDOW_NORMAL)
    cv2.imshow("Debug Image", debug_image)
    cv2.imshow("Debug Region", debug_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #################
