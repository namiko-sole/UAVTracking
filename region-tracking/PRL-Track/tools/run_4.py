import os
import cv2
import torch
import sys
import time
sys.path.append("./")
import numpy as np
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.prl_tracker import PRLTrack
from pysot.utils.model_load import load_pretrain

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_tracker(config_path, model_path):
    """加载 PRL-Track 跟踪器"""
    cfg.merge_from_file(config_path)
    model = ModelBuilder()
    model = load_pretrain(model, model_path).cuda().eval()
    tracker = PRLTrack(model)
    return tracker

def select_target(frame):
    """手动选择初始目标区域"""
    bbox = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")
    return bbox

def track_video(video_path, tracker, config_path, model_path):
    """对单个视频进行目标跟踪"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    # 设置视频编解码器，FourCC代码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(1920/2)
    height = int(1080/2)
    out = cv2.VideoWriter('output_5.mp4', fourcc, 20.0, (width, height))

    #init_bbox_selected = False  # 初始化目标选择标志

    fps = 0  # 初始化FPS
    start_time = time.time()  # 获取当前时间
    frame_count = 0  # 初始化帧计数器
    x = width/2
    y = height/2
    w = 100
    h = 100
    init_bbox = [x, y, w, h]  # 转为中心点坐标格式
    init = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束")
            break

        frame = cv2.resize(frame, (width, height))
        if init == False:
            tracker.init(frame, init_bbox)  # 初始化跟踪器
            init = True
        
        # 如果目标尚未选择，并且用户按下了“P”键
        if cv2.waitKey(100) == ord("p"):
            init_bbox = select_target(frame)  # 手动选择初始目标
            x, y, w, h = init_bbox
            
            init_bbox = [x, y, w, h]  # 转为中心点坐标格式
            tracker.init(frame, init_bbox)  # 初始化跟踪器
            #init_bbox_selected = True  # 设置目标选择标志

        # 如果目标已选择，进行跟踪
        
        outputs = tracker.track(frame)
        pred_bbox = outputs["bbox"]
        x, y, w, h = map(int, pred_bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
                frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow("Tracking", frame)

        # 按 ESC 退出
        if cv2.waitKey(1) == 27:
            break
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 配置文件路径
    config_path = "./experiments/config.yaml"
    model_path = "./tools/snapshot/best.pth"

    input = input("请输入1或2：")
    # 视频文件路径
    if(input == "1"):
        video_path = "/home/niuhy24/PRL-Track/load_1.mp4"  # 替换为你的视频文件路径
    elif(input == "2"):
        video_path = "/home/niuhy24/object-tracker-master/docs/dj.MP4"
    else:
        print("请输入1或2")
    # 加载跟踪器
    tracker = load_tracker(config_path, model_path)

    # 在视频中进行跟踪
    track_video(video_path, tracker, config_path, model_path)
