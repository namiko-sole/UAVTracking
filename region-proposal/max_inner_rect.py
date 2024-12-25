import numpy as np
import cv2
import time

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

    return [edge[3], edge[2], edge[1], edge[0]]
if __name__ == '__main__':
    # 需要引入一个python库shapely用来寻找轮廓的质心
    contour = np.array([[ 301,  300],[ 300,  301],[ 300,  626],[ 301,  627],[1071,  627],[1072,  628],[1072, 637],[1073,  638],[1251,638],
     [1252, 637],[1252,  628],[1253,  627],[1297,  627], [1298,  626],[1298,  625],[1299,  624],[1788, 624],[1790,  622],[1792,622],
     [1793,  621],[2244,  621],[2245,  622],[2247,  622],[2249,  624],[3161,  624],[3162, 625],[3162,  636],[3163,  637],[3433,637],
     [3434,  636],[3434,  625],[3435,  624],[4285,  624],[4286,  623],[4286,  378],[4285,  377],[4179,  377],[4178,  376],[4178,322],
     [4177,  321],[4168,  321],[4167,  320],[4167,  310],[4166,  309],[4030,  309],[4029,  310],[4029,  320],[4028,  321],[3811,  321],
     [3810,  320],[3810, 310],[3809,  309],[3673,  309],[3672,  310],[3672,  320],[3671, 321],[3662,  321],[3661,  322],[3661,  376],
     [3660,  377],[3396, 377],[3394,  375],[3230,  375],[3228,  377],[2964,  377],[2963 , 376],[2963 , 322],[2962,  321],[2953,  321],
     [2952 , 320],[2952 , 310],[2951,  309],[2816 , 309],[2815 , 310],[2815 , 320],[2814 , 321],[2596 , 321],[2595 , 320],[2595 , 310],
     [2594 , 309],[2458,  309],[2457 , 310],[2457 , 320],[2456 , 321],[2447 , 321],[2446 , 322],[2446 , 376],[2445 , 377],[2181 , 377],
     [2179 , 375],[2016 , 375],[2014 , 377],[1750 , 377],[1749 , 376],[1749 , 322],[1748 , 321],[1738 , 321],[1737 , 320],[1737 , 310],
     [1736,  309],[1601 , 309],[1600 , 310],[1600 , 320],[1599 , 321],[1381 , 321],[1380 , 320],[1380 , 310],[1379 , 309],[1243 , 309],
     [1242 , 310],[1242 , 320],[1241 , 321],[1232 , 321],[1231 , 322],[1231,  376],[1230 , 377],[ 966 , 377],[ 964 , 375],[ 801 , 375],
     [ 799 , 377],[ 535 , 377],[ 534 , 376],[ 534 , 301],[ 533 , 300]])

    from shapely.geometry import Polygon
    center = Polygon(contour).centroid
    center = list(map(int, [center.x, center.y]))
    img = np.zeros((1000, 4600))
    cv2.fillPoly(img, [contour.reshape(-1, 1, 2).astype('int')], 255)

    start_time = time.time()
    res = find_max_inner_rectangle(img, center)
    print(time.time()-start_time)


    box = [np.array([[res[0], res[1]],
                     [res[0], res[3]],
                     [res[2], res[3]],
                     [res[2], res[1]]])]
    rgb_img = np.zeros((1000, 4600, 3))
    cv2.fillPoly(rgb_img, [contour.reshape(-1, 1, 2).astype('int')], (255,255,255))
    cv2.drawContours(rgb_img, box, 0, (0,0,255), 3)
    cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Region Proposal Test", rgb_img)
    # cv2.imshow("Region Proposal Edges", quan_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 得出来的结果是找到的最大的内接矩形的bbox,包括左上点和右下点
