



import cv2
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5 import QtGui, QtWidgets, Qt, QtCore
import sys
from gui import Ui_MainWindow

import time
import os

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer

from workerVideoThread import Worker1_video
from AlgorithmModule import Algorithm
from timeThread import RepeatedTimer

import threading as th

import random

#......................................................................
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		super(MyApp, self).__init__()
		self.setupUi(self)
		self.dir = os.getcwd()
		self.startBtn.setEnabled(True)#disable open_btn
		self.pauseBtn.setEnabled(False)#enable close_btn

		self.videoSize = self.video_lbl.height(), self.video_lbl.width() #self.video_lbl.size()

		self.workerVideoThread = Worker1_video(self, self.videoSize,self.video_lbl)
		self.workerVideoThread.ImageUpdate.connect(self.ImageUpdateSlot)
		self.pauseBtn.clicked.connect(self.pauseVideo)
		self.startBtn.clicked.connect(self.startVideo)
		self.images = []
		self.readImages((self.dir + "/images/fruits_vegatables/BG"))
		self.video_lbl.setStyleSheet("background - color: transparent;")

#		self.workerSerialThread.message.connect(self.readBytesSlot)
		#t1 = timerEvent(2.0)
		#t1.startt()

		#self.t = th.Timer(2.0, self.func)
		#self.t.start()
		self.rt = RepeatedTimer(0.7, self.hello, "World")  # it auto-starts, no need of rt.start()
		#self.x = 0

	def hello(self,name):
		#r,g,b = random.randint(0,255), random.randint(0,255), random.randint(0,255)
		#str="color: rgba({}, {}, {}, 255)".format(r,g,b)
		#self.label_3.setStyleSheet(str)
		#self.x = (self.x + 15) % self.frameGeometry().width()
		#self.label_3.setGeometry(QtCore.QRect(self.x, 10, 301, 51))
		#if self.videoMode != "on":
		self.video_lbl.setScaledContents(True)
		self.video_lbl.setPixmap(self.images[random.randint(1,len(self.images)-1)])

	def readImages(self, s):
		for root, dirr, filename in os.walk(s):
			for fn in filename:
				pix = QtGui.QPixmap(root+"/"+fn)
				#pix = pix.scaled(200,200)
				self.images.append(pix)
		print("bg images uploaded" + str(len(self.images)))

	def ImageUpdateSlot(self, image):
		#self.video_lbl.setPixmap(QPixmap.fromImage(image))
		self.video_lbl.setPixmap(image)
		prodName, prodPrice = Worker1_video.getProduct(self)
		self.typeLbl.setText(prodName)
		self.priceLbl.setText(str(prodPrice))

		path = (self.dir+"/images/fruits_vegatables/{}.jpg").format(prodName)
		prodImg = QtGui.QPixmap(path)
		self.productImg.setPixmap(prodImg)
		#self.durationLbl.setText(str(Worker1_video.getDuaration(self)))

	def startVideo(self):
		self.rt.stop()
		self.workerVideoThread.setIdCamera(0)
		self.startBtn.setEnabled(False)  # disable startBtn
		self.pauseBtn.setEnabled(True)  # enable stopBtn
		#print("before start")
		self.workerVideoThread.start()
		#print("after start")

	def pauseVideo(self):
		self.startBtn.setEnabled(True)  # disable startBtn
		self.pauseBtn.setEnabled(False)  # enable stopBtn
		self.workerVideoThread.stop()
		self.rt.start()

	def exitApp(self):
		ret = QMessageBox.question(self, 'Exit', "Are you sure !!",QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if ret == QMessageBox.Yes:
			self.close()

	def printStatus(self, s):
		ret = QMessageBox.information(self, 'Error', s, QMessageBox.Yes)
#......................................................................

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	Form = MyApp()
	Form.show()
	sys.exit(app.exec_())
	self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
	self.SetAttribute(QtCore.Qt.WA_TranslucentBackground)
