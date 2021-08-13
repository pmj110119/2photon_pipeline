#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-21 14:16:48
# @Author  : Lewis Tian (chtian@hust.edu.cn)
# @Link    : https://lewistian.github.io
# @Version : Python3.7

import time,os,sys,cv2,qdarkstyle
import numpy as np
#import QMutex

from PIL import Image
from numpy.core.fromnumeric import resize
from PyQt5 import QtCore, QtGui, QtWidgets

from libs import *


class Denoise():
    def run(img,func=gaussion):
        return func(img)




class DenoiseProcess(QThread):
    # 视频处理线程
    img_res = pyqtSignal(np.ndarray)  # 一副图像完成处理后，以信号量的方式传出去
    def __init__(self,model, fps=100):
        super(DenoiseProcess,self).__init__()
        self.running = False
        self.stop_flag =False
        self.model = model
        self.fps_max = fps
    def setCap(self,cap):
        # 传入cap对象
        self.cap = cap
    def is_running(self):
        return self.running
    def stop(self):
        self.stop_flag = True


    def run(self):
        self.running = True
        ref, frame = self.cap.read()
        fps = 0.0
        while ref:
            if self.stop_flag:  # 提前中止
                self.stop_flag = False
                break
            t1 = time.time()
            # 处理一帧图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(self.model.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t2 = time.time()
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 发射信号量
            self.img_res.emit(frame)
            # 读取新的一帧
            ref, frame = self.cap.read()
            
        self.running = False


       
