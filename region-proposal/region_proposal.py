import cv2
import numpy as np
import scipy
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

    # return [edge[3], edge[2], edge[1], edge[0]]
    # return np.array([[edge[0], edge[1]],[edge[0], edge[3]], [edge[2], edge[3]],[edge[2], edge[1]]])
    return np.array([[edge[3], edge[2]],[edge[3], edge[0]], [edge[1], edge[0]],[edge[1], edge[2]]])


def get_obvious_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)

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
        # rect = cv2.minAreaRect(cnt)
        # width, height = rect[1][0], rect[1][1]
        # wh_scale = width / (height+1e-5)
        # area = width * height
        if area < 2000 or area > 200000: continue
        # if wh_scale<0.2 or wh_scale>3: continue

        # M = cv2.moments(cnt)
    
        # # 使用矩计算中心点
        # if M['m00'] != 0:
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])
            
        #     if plain_region[cy, cx] !=0:
        #         x,y,w,h = cv2.boundingRect(cnt)

        #         while True:
        #             cx = np.random.randint(x, x+w)
        #             cy = np.random.randint(y, y+h)
        #             if plain_region[cy, cx] == 0:
        #                 break
        # rect = find_max_inner_rectangle(~plain_region, (cx, cy))

        row, column = 5,5
        x,y,w,h = cv2.boundingRect(cnt)
        # gap_x, gap_y = w//row, h//column
        cx = sorted(np.linspace(x, x+w, row, dtype=np.uint32), key=lambda c:abs(c-(x+w)//2))
        cy = sorted(np.linspace(y, y+h, column, dtype=np.uint32), key=lambda c:abs(c-(x+w)//2))
        sample_centers = np.array(np.meshgrid(cx, cy)).reshape(2,-1).swapaxes(0,1)
        for center in sample_centers:
            cv2.circle(image, (center[0], center[1]), 5, (255, 0, 0), -1)
            if plain_region[center[1], center[0]] != 0: continue
            rect = find_max_inner_rectangle(~plain_region, center)
            cv2.drawContours(image, [rect], 0, (0,255,0), 3)
            cv2.rectangle(plain_region, rect[0], rect[2], 255, -1)
        
        # 在图像上绘制中心点
        # cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

        cv2.drawContours(image, [cnt], 0, (0,0,255), 3)
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
    image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\WeChat_20241220211216.png",dtype=np.uint8),cv2.IMREAD_COLOR)
    # image = cv2.imdecode(np.fromfile(r"E:\github\UAVTracking\test_data\cut_video.mp4_20241223_133516.114.jpg",dtype=np.uint8),cv2.IMREAD_COLOR)

    # get_obvious_region(image)
    # get_goods_region(image, 5)
    get_plain_region(image)
    
    pass
