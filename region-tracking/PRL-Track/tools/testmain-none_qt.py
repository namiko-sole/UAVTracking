import sys
import threading
from datetime import datetime
from time import sleep
import cv2
sys.path.append("./")
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.prl_tracker import PRLTrack
from pysot.utils.model_load import load_pretrain


from simulator import DroneSimulator
import multiprocessing

map_path = "/home/niuhy24/PRL-Track/pid.jpg"  # 地图文件路径
window_range = 1920/2,1080/2  # 滑动窗口范围
tracking_area = [(2300,3400 ), (2400, 3500)]  # 追踪区域
ship_speed = [400.0, 0.0]  # 船的速度 (x, y)
pid_params = (0.5, 0.1, 0.2)  # PID控制参数 (kp, ki, kd)
map1 = cv2.cvtColor(cv2.imread(map_path), cv2.COLOR_BGR2RGB)

class pyui():
    def __init__(self):
        super().__init__()
        self.drone = DroneSimulator(map_path, window_range, tracking_area, ship_speed, pid_params)
        self.speed=self.drone.update0()
        self.frameList=[]
        self.begintime=0
        self.num=0
        self.init = False
        self.tracker=self.init_tracker()
        self.lasttime=0
    def load_tracker(self,config_path, model_path):
        """加载 PRL-Track 跟踪器"""
        cfg.merge_from_file(config_path)
        model = ModelBuilder()
        model = load_pretrain(model, model_path).cuda().eval()
        tracker = PRLTrack(model)
        return tracker

    def init_tracker(self):
        config_path = "./experiments/config.yaml"
        model_path = "./tools/snapshot/best.pth"
        return self.load_tracker(config_path, model_path)
    def track_video(self,frame):
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

    def update(self,dt):
        frame = self.speed["imgmap"].copy()
        #y, x = self.speed["imgmap"].shape[:-1]
        #self.frameList.insert(len(frame),frame)

        #连接追踪
        x0,y0,x, y, w, h=self.track_video(frame)
        #print(dt)
        self.speed = self.drone.updateNew(dt,x0,y0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
                frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        cv2.imwrite('./result_2/img'+str(self.num)+'.png', frame)
        self.num+=1

    def update_new(self):
        self.begintime = datetime.now()
        self.lasttime = datetime.now()
        while True:
            nowtime = datetime.now()
            dt = (nowtime - self.lasttime).total_seconds()
            self.lasttime=nowtime
            self.update(dt)

            sleep(0.1)
            time_diff = (nowtime - self.begintime).total_seconds()
            if (time_diff > 120 or self.speed["plane_positionx"] - self.speed["ship_positionx"]
                    +self.speed["plane_positiony"] - self.speed["ship_positiony"]<1e-5):
                break




if __name__ == '__main__':
    pyui=pyui()
    pyui.update_new()
