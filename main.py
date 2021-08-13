#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time,os,sys,cv2,qdarkstyle
import numpy as np
from PIL import Image
#import QMutex
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets

from libs import *


Denoise_func = {
    'gaussian': gaussianBlur,
    'mean': meanBlur,
    'median': medianBlur
}

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("./assets/main.ui", self)   # 以文件形式加载ui界面
  
        self.img_list = []
        self.openImg.triggered.connect(self.open_image)




        self.denoise_func = 'gaussian'
        self.Denoise_size = {
            'gaussian': self.slider_gaussian,
            'mean': self.slider_mean,
            'median': self.slider_median
        }


        self.button_gaussian.clicked.connect(self.set_gaussion)
        self.button_mean.clicked.connect(self.set_mean)
        self.button_median.clicked.connect(self.set_median)


        self.frame_slider.valueChanged.connect(self.frame_change)
        self.slider_gaussian.valueChanged.connect(self.frame_change)
        self.slider_mean.valueChanged.connect(self.frame_change)
        self.slider_median.valueChanged.connect(self.frame_change)

        
    def open_image(self):
        openfile = QFileDialog.getOpenFileNames(self, '选择文件', '', 'image files(*.tif)')[0]
        if openfile:
            ret, images = cv2.imreadmulti(openfile[0], [], cv2.IMREAD_ANYCOLOR)
            if ret:
                self.img_list = images
                self.plot(images[0])
                self.frame_slider.setMaximum(len(images)-1)
                self.frame_slider.setValue(0)

    def frame_change(self):
        if self.img_list:
            id = self.frame_slider.value()
            self.plot(self.img_list[id])



    def set_gaussion(self):
        self.denoise_func = 'gaussian'
        self.frame_change()
    def set_mean(self):
        self.denoise_func = 'mean'
        self.frame_change()
    def set_median(self):
        self.denoise_func = 'median'
        self.frame_change()



    def plot(self,img):
        """
        将图像显示在QLabel上
        """
        img = cv2.resize(img,(512,512))
        img_denoise = Denoise_func[self.denoise_func](img, self.Denoise_size[self.denoise_func].value())
        #img_denoise = nonLocalMeans(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.curve_origin.setPixmap(QPixmap.fromImage(img))

        img_denoise = cv2.cvtColor(img_denoise, cv2.COLOR_RGB2BGR)
        img_denoise = QImage(img_denoise.data, img_denoise.shape[1], img_denoise.shape[0], QImage.Format_RGB888)
        self.curve_denoise.setPixmap(QPixmap.fromImage(img_denoise))


    # 单击“检测图片按钮”的槽函数
    def detect_img(self):
        if self.cap_thread.is_running():
            self.cap_thread.stop()
        file = QFileDialog.getOpenFileName(self, '选择图片', '', 'images(*.png; *.jpg);;*')[0]
        if not file:
            return
        img = cv2.imread(file)
        #检测图片
        r_image = self.model.detect_image(Image.fromarray(np.uint8(img)))
        img = np.array(r_image)
        # 显示图片
        self.plot(img)


 
        






if __name__ == '__main__':
    app = QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    app.setStyleSheet(stylesheet)
    w = GUI()
    w.show()
    sys.exit(app.exec_())