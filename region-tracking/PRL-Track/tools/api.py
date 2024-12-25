import sys
sys.path.append('.')
from datetime import datetime

import cv2
import time
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.prl_tracker import PRLTrack
from pysot.utils.model_load import load_pretrain


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.num=0

    def calculate(self, error, dt):
        self.integral += error #* dt
        derivative = (error - self.previous_error) #/ dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output


class UAVTracking:
    """无人机自动跟踪系统"""
    def __init__(self, config_path, model_path, pid_params=(0.5,0.0,0.0), debug=False, max_speed_x=100, max_speed_y=100):
        self.debug = debug
        self.dt = 0
        self.lasttime = 0
        self.init = False
        self.step=0
        self.max_speed_x = max_speed_x
        self.max_speed_y = max_speed_y

        self.last_frame = None
        self.last_bbox = None
        self.relative_bbox = None
        self.simulator_bbox = None
        self.width = 0
        self.heigt = 0

        self.pid_x = PIDController(*pid_params)
        self.pid_y = PIDController(*pid_params)

        self.tracker = self.__load_tracker(config_path, model_path)
    
    def __load_tracker(self,config_path, model_path):
        """加载 PRL-Track 跟踪器"""
        print("正在加载跟踪器...")
        strat_time = time.time()
        cfg.merge_from_file(config_path)
        model = ModelBuilder()
        model = load_pretrain(model, model_path).cuda().eval()
        tracker = PRLTrack(model)
        print(f"加载跟踪器耗时{time.time()-strat_time:.2f}秒。")
        return tracker
    
    def __get_speed(self,dt,errorx,errory):
        # print(f"dt: {dt} | error: {errorx}, {errory} ")
        return self.pid_x.calculate(errorx,dt), self.pid_y.calculate(errory,dt)
    
    def __track_once(self,frame):
        if not self.init:
            init_bbox = self.region_proposal(frame)[0]
            self.tracker.init(frame, init_bbox)  # 初始化跟踪器
            self.init = True

        # 如果目标已选择，进行跟踪
        outputs = self.tracker.track(frame)
        pred_bbox = outputs["bbox"]
        # x, y, w, h = map(int, pred_bbox)
        return map(int, pred_bbox)

    def track(self,frame):
        if self.width==0 and self.heigt == 0:
            self.width, self.heigt = frame.shape[1], frame.shape[0]
            bbox_wh = 100
            # self.simulator_bbox = [self.width//2-bbox_wh//2, self.heigt//2-bbox_wh//2, bbox_wh, bbox_wh]
            self.simulator_bbox = [0, 0, bbox_wh, bbox_wh]
        dt = (datetime.now() - self.lasttime).total_seconds() if self.lasttime!=0 else 0
        x, y, w, h = self.__track_once(frame)
        self.last_bbox = [x,y,w,h]

        pred_x, pred_y = self.__get_bbox_center(self.last_bbox)
        if self.debug:
            target_x, target_y = self.__get_bbox_center(self.simulator_bbox)
        else:
            target_x, target_y = self.width//2, self.heigt//2
        
        if self.relative_bbox is not None:
            pred_x += self.relative_bbox[0]
            pred_y += self.relative_bbox[1]

        err_x, err_y = pred_x-target_x, pred_y-target_y
        speed_x, speed_y = self.__get_speed(dt, err_x, err_y)
        speed_x = min(speed_x, self.max_speed_x) if speed_x>0 else max(speed_x, -self.max_speed_x)
        speed_y = min(speed_y, self.max_speed_y) if speed_y>0 else max(speed_y, -self.max_speed_y)

        if self.debug:
            self.simulator_bbox[0] = int(self.simulator_bbox[0]+speed_x)
            self.simulator_bbox[1] = int(self.simulator_bbox[1]+speed_y)
            print(f"steps: {self.step} | dt: {dt} | error: {err_x}, {err_y} | speed_x: {speed_x} | speed_y: {speed_y}")

        self.lasttime = datetime.now()
        self.last_frame = frame
        self.step += 1
        return  speed_x, speed_y, self.last_bbox #返回值为x方向和y方向速度
    
    def set_track_bbox(self, frame, bbox):
        self.tracker.init(frame, bbox)
        if self.relative_bbox is not None:
            _relative_bbox = self.get_relative_bbox()
            self.last_bbox = bbox
            self.set_relative_bbox(frame, _relative_bbox)
        else:
            self.last_bbox = bbox

    def set_relative_bbox(self, frame, bbox):
        ox, oy = self.__get_bbox_center(self.last_bbox)
        tx, ty = self.__get_bbox_center(bbox)
        self.relative_bbox = [tx-ox, ty-oy, bbox[2], bbox[3]]

    def get_relative_bbox(self):
        ox, oy = self.__get_bbox_center(self.last_bbox)
        tx, ty, tw, th = self.relative_bbox
        return [tx+ox-tw//2, ty+oy-th//2, tw, th]

    def reset_relative_bbox(self):
        self.relative_bbox = None

    def get_debug_iamge(self):
        _image = self.last_frame.copy()
        last_bbox = self.__get_bbox_points(self.last_bbox)
        cv2.drawContours(_image, [np.array(last_bbox)], 0, (0,0,255), 3)
        if self.relative_bbox is not None:
            relative_bbox = self.__get_bbox_points(self.get_relative_bbox())
            cv2.drawContours(_image, [np.array(relative_bbox)], 0, (0,255,0), 3)
        if self.debug:
            debug_bbox = self.__get_bbox_points(self.simulator_bbox)
            cv2.drawContours(_image, [np.array(debug_bbox)], 0, (255,0,0), 3)
        return _image
    
    def __get_bbox_points(self, xywh):
        x, y, w, h = xywh
        x1y1 = [x,y]
        x2y1 = [x+w-1, y]
        x2y2 = [x+w-1, y+h-1]
        x1y2 = [x, y+h-1]
        return [x1y1, x2y1, x2y2, x1y2]

    def __get_bbox_center(self, xywh):
        x, y, w, h = xywh
        return [x+w//2, y+h//2]

    def region_proposal(self, frame):
        # height, width = 1080//2,1920//2
        # x = width / 2
        # y = height / 2
        x = self.width // 2 - 50
        y = self.heigt // 2 - 50
        w = 100
        h = 100
        bbox = [x, y, w, h]  # 转为中心点坐标格式
        return [bbox]


class VideoData:
    def __init__(self, video_path):
        self.index = 0
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("视频文件读取失败，请检查视频路径是否正确。")
        self.init_frmaes()
        
    def init_frmaes(self):
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            # if len(self.frames)>=200: break
            if not ret:
                print(f"视频读取完成。共读取到{len(self.frames)}个图像。")
                break
            self.frames.append(frame)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.frames):
            raise StopIteration
        else:
            frame = self.frames[self.index]
            self.index += 1
            return frame

if __name__ == "__main__":
    cv2.namedWindow("Debug Image", cv2.WINDOW_NORMAL)
    # 指定模型配置文件与模型文件
    config_path = "experiments/config.yaml"
    model_path = "tools/snapshot/best.pth"

    # 初始化跟踪模型
    tracker = UAVTracking(config_path, model_path, debug=True)

    # 获取跟踪图像数据
    frames = VideoData(r"E:\github\UAVTracking\test_data\cut_video.mp4")
    
    # 跟踪并返回速度
    for frame in frames:
        speed_x, speed_y, bbox = tracker.track(frame)
        
        debug_image = tracker.get_debug_iamge()

        ########## Debug可视化 ##########
        cv2.imshow("Debug Image", debug_image)

        if cv2.waitKey(3) == ord("p"):  # 按下p键来选择稳定区域
            bbox = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Target")
            tracker.set_track_bbox(frame, bbox)  # 设置稳定区域跟踪框
        
        if cv2.waitKey(3) == ord("o"):  # 按下o键来选择投放区域
            bbox = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Target")
            tracker.set_relative_bbox(frame, bbox)  # 设置投放区域跟踪框

        if cv2.waitKey(1) == 27:  # 按下Esc键退出
            break
        ################################

