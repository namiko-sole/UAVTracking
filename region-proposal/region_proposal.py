import cv2
import numpy as np
import scipy


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


def get_obvious_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
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
        if area > 20000: continue
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
        cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Region Proposal Edges", cv2.WINDOW_NORMAL)
        cv2.imshow("Region Proposal Test", _image)
        cv2.imshow("Region Proposal Edges", segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 5, 50)
    edges[:150, :] = 0
    edges[-400:, :] = 0

    # edges[:, :400] = 0
    # edges[:, -400:] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # rect = cv2.minAreaRect(cnt)
        # width, height = rect[1][0], rect[1][1]
        # wh_scale = width / (height+1e-5)
        # area = width * height
        if area < 1000: continue
        # if area > 20000: continue
        # if wh_scale<0.2 or wh_scale>3: continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxs.append(box)

        cv2.drawContours(image, [cnt], 0, (0,0,255), 3)
    cv2.namedWindow("Region Proposal Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Region Proposal Test", image)
    cv2.imshow("Region Proposal Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxs


if __name__ == "__main__":
    # image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\WeChat_20241220211216.png",dtype=np.uint8),cv2.IMREAD_COLOR)
    image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\cut_video.mp4_20241223_133516.114.jpg",dtype=np.uint8),cv2.IMREAD_COLOR)
    
    # get_obvious_region(image)
    # get_goods_region(image, 10)
    get_plain_region(image)
    pass
