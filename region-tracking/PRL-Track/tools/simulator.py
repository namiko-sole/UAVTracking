from datetime import datetime
from time import sleep

import numpy as np
import random
import cv2  # OpenCV库用于图像操作


# PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

# 模拟器类
class DroneSimulator:
    def __init__(self, map_path, window_size, tracking_area, ship_speed, pid_params):
        # 加载地图
        self.map = cv2.imread(map_path)
        self.lasttime = datetime.now()
        if self.map is None:
            raise ValueError("地图文件加载失败，请检查路径是否正确。")
        self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        self.map_size = self.map.shape[:2][::-1]  # 地图大小 (宽, 高)
        self.initPosition=[3457,2344,]
        # 随机生成滑动窗口大小
        self.window_size = window_size

        # 随机生成滑动窗口的初始左上角位置
        self.window_position = self.initPosition.copy()

        # 初始化无人机和船的位置
        self.drone_position = np.array(self.window_position, dtype=float)
        self.ship_position = np.array([0,0],dtype=float)

        # 初始化无人机速度和船速度
        self.drone_speed = np.array([0.0, 0.0])
        self.ship_speed = np.array(ship_speed)

        # 初始化PID控制器
        self.pid_x = PIDController(*pid_params)
        self.pid_y = PIDController(*pid_params)

        # 记录误差历史
        self.error_history = []
    def update0(self):
        speed = {"ship_speedx": self.ship_speed[0], "ship_speedy": self.ship_speed[1],
                 "plane_speedx": self.drone_speed[0], "plane_speedy": self.drone_speed[1],
                 "ship_positionx": self.ship_position[0], "ship_positiony": self.ship_position[1],
                 "plane_positionx": self.drone_position[0], "plane_positiony": self.drone_position[1],
                 #"imgmap": self.map,
                 "imgmap": self.get_window_view()
                 }
        return speed
    def update(self):
        """
        根据船和无人机的速度更新位置，并调整滑动窗口位置。
        """
        # 计算误差

        # dt = (datetime.now() - self.lasttime).total_seconds()
        dt=1
        self.lasttime=datetime.now()
        error_x = (self.ship_position[0]-self.drone_position[0]+self.initPosition[0])
        error_y = (self.ship_position[1]-self.drone_position[1]+self.initPosition[1])
        self.error_history.append((error_x, error_y))
        print(self.ship_position, self.drone_position, self.initPosition)
        print(error_x, error_y)
        # 更新无人机速度（PID控制）
        self.drone_speed[0] = self.pid_x.calculate(error_x, dt)
        self.drone_speed[1] = self.pid_y.calculate(error_y, dt)

        # 更新无人机和船的位置
        self.drone_position += self.drone_speed * dt
        self.ship_position += self.ship_speed * dt

        # 更新滑动窗口位置
        self.window_position[0] = int(self.drone_position[0]-self.ship_position[0])
        self.window_position[1] = int(self.drone_position[1]-self.ship_position[1])
        speed = {"ship_speedx": self.ship_speed[0], "ship_speedy": self.ship_speed[1],
                 "plane_speedx": self.drone_speed[0], "plane_speedy": self.drone_speed[1],
                 "ship_positionx": self.ship_position[0], "ship_positiony": self.ship_position[1],
                 "plane_positionx": self.drone_position[0], "plane_positiony": self.drone_position[1],
                 "imgmap": self.map,
                 "imgmap2": self.get_window_view()
                 }
        return speed

    def updateNew(self,dt,error_x,error_y):
        """
        根据船和无人机的速度更新位置，并调整滑动窗口位置。
        """
        # 更新无人机速度（PID控制）
        self.drone_speed[0] = self.pid_x.calculate(error_x, dt)
        self.drone_speed[1] = self.pid_y.calculate(error_y, dt)

        # 更新无人机和船的位置
        print("dt:",dt)
        print("error:",error_x, error_y)
        #print("speed:",self.drone_speed)
        #print("position_1:",self.drone_position, self.ship_position)
        self.drone_position += self.drone_speed * dt
        self.ship_position += self.ship_speed * dt
        #print("position_2:",self.drone_position, self.ship_position)
        # 更新滑动窗口位置
        self.window_position[0] = int(self.drone_position[0]-self.ship_position[0])
        self.window_position[1] = int(self.drone_position[1]-self.ship_position[1])
        speed = {"ship_speedx": self.ship_speed[0], "ship_speedy": self.ship_speed[1],
                 "plane_speedx": self.drone_speed[0], "plane_speedy": self.drone_speed[1],
                 "ship_positionx": self.ship_position[0], "ship_positiony": self.ship_position[1],
                 "plane_positionx": self.drone_position[0], "plane_positiony": self.drone_position[1],
                 "imgmap": self.get_window_view()
                 }
        return speed

    def get_window_view(self):
        """
        返回当前滑动窗口视图。
        如果窗口超出地图边界，打印提示并返回None。
        """
        x, y = self.window_position
        size = self.window_size
        size = (int(size[0]), int(size[1]))
        #print(x,y,size)
        #print(1)
        if x < 0 or y < 0 or x + size[0] > self.map_size[0] or y + size[1] > self.map_size[1]:
            raise ValueError("跟随失败")
        return self.map[y:y+size[1], x:x+size[0]]


if __name__ == '__main__':
    # 示例参数
    map_path = "C:\\Users\\gaoxi\\Desktop\\DJI_20240605135922_0029_W.jpg"  # 地图文件路径
    window_range = (200, 1000)  # 滑动窗口范围
    tracking_area = [(2000, 2000), (3000, 3000)]  # 追踪区域
    ship_speed = [3.0, 4.0]  # 船的速度 (x, y)
    pid_params = (0.5, 0.1, 0.2)  # PID控制参数 (kp, ki, kd)

# 初始化应用

