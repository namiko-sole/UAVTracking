import sys
import threading
from datetime import datetime
import time
import cv2
sys.path.append("./")
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.prl_tracker import PRLTrack
from pysot.utils.model_load import load_pretrain


from simulator_new import DroneSimulator
import multiprocessing

map_path = r"E:\github\UAVTracking\region-tracking\PRL-Track\pid.jpg"  # 地图文件路径
window_range = 1920/2,1080/2  # 滑动窗口范围
tracking_area = [(2300,3400 ), (2400, 3500)]  # 追踪区域
ship_speed = [100.0, 0.0]  # 船的速度 (x, y)
pid_params = (0.5, 0.1, 0.2)  # PID控制参数 (kp, ki, kd)
map1 = cv2.cvtColor(cv2.imread(map_path), cv2.COLOR_BGR2RGB)

def cal_fps(fps_start_time):
    if fps_start_time is None:
        fps_start_time = time.time()
        fps = 0
    else:
        cur_time = time.time()
        fps = 1/(cur_time-fps_start_time)
        print('fps:',fps)
        fps_start_time=cur_time
    return fps,fps_start_time

class pyui():
    def __init__(self):
        super().__init__()
        self.drone = DroneSimulator(map_path, window_range, tracking_area, ship_speed, pid_params)
        self.speed=self.drone.update0()
        self.frameList=[]
        self.begintime=0
        self.lasttime=0


    def update_new(self):
        
        self.begintime = datetime.now()
        self.lasttime = datetime.now()
        while True:
            start_time = time.time()
            nowtime = datetime.now()
            frame = self.speed["imgmap"]
            self.speed=self.drone.updateNew(frame)

            time_diff = (nowtime - self.begintime).total_seconds()
            if (time_diff > 120 or self.speed["plane_positionx"] - self.speed["ship_positionx"]
                    +self.speed["plane_positiony"] - self.speed["ship_positiony"]<1e-5):
                break
            cal_fps(start_time)
            




if __name__ == '__main__':
    pyui=pyui()
    pyui.update_new()
