from datetime import datetime

import cv2

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
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class Realtime:
    def __init__(self,pid_params):
        self.pid_x = PIDController(*pid_params)
        self.pid_y = PIDController(*pid_params)
        self.tracker = self.__init_tracker()
        self.dt = 0
        self.lasttime = 0
        self.init = False
        self.num=0

    def __init_tracker(self):
        config_path = "experiments/config.yaml"
        model_path = "tools/snapshot/best.pth"
        return self.__load_tracker(config_path, model_path)
    def __load_tracker(self,config_path, model_path):
        """加载 PRL-Track 跟踪器"""
        cfg.merge_from_file(config_path)
        model = ModelBuilder()
        model = load_pretrain(model, model_path).cuda().eval()
        tracker = PRLTrack(model)
        return tracker
    def __getspeed(self,dt,errorx,errory):
        print("dt:",dt)
        print("error:",errorx, errory)
        return self.pid_x.calculate(errorx,dt), self.pid_y.calculate(errory,dt)
    def __track_video(self,frame):
        height, width = 1080//2,1920//2
        x = width / 2
        y = height / 2
        w = 100
        h = 100
        init_bbox = [x, y, w, h]  # 转为中心点坐标格式

        frame = cv2.resize(frame, (width, height))
        if not self.init:
            self.tracker.init(frame, init_bbox)  # 初始化跟踪器
            self.init = True

            # 如果目标已选择，进行跟踪
        outputs = self.tracker.track(frame)
        pred_bbox = outputs["bbox"]
        x, y, w, h = map(int, pred_bbox)
        return x+w/2-1920/4, y+h/2-1080/4,x, y, w, h

    def track(self,frame):
        if self.lasttime==0:
            self.lasttime =datetime.now()
        dt = (datetime.now() - self.lasttime).total_seconds()
        self.lasttime = datetime.now()
        x0, y0, x, y, w, h = self.__track_video(frame)
        x,y=self.__getspeed(dt, x0, y0)
        return  x,y  #返回值为x方向和y方向速度

    def trackWithRect(self,frame):           #仅供测试使用，实际功能与track相同
        frame = frame.copy()
        if self.lasttime==0:
            self.lasttime =datetime.now()
        dt = (datetime.now() - self.lasttime).total_seconds()
        self.lasttime = datetime.now()
        x0, y0, x, y, w, h = self.__track_video(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        cv2.imwrite(r'E:\github\UAVTracking\test_data\tmp\img' + str(self.num) + '.png', frame)
        self.num += 1
        x,y=self.__getspeed(dt, x0, y0)
        return x,y,dt