
import cv2
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5 import QtGui, QtWidgets, Qt, QtCore
import sys
from gui import Ui_MainWindow


import time
from PyQt5.QtCore import QIODevice
import os


from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QIODevice, QThread, pyqtSignal, pyqtSlot, QTimer

from AlgorithmModule import Algorithm
from timer import Timer
from imageEnhancment import imageEnhancment
import numpy as np


class Worker1_video(QThread, Algorithm):

	#ImageUpdate = pyqtSignal(QImage)
	ImageUpdate = pyqtSignal(QtGui.QPixmap)


	def __init__(self, parent, video_size,video_lbl):
		QThread.__init__(self, parent=parent)
		self.video_size = video_size
		self.video_lbl = video_lbl
		self.ThreadActive = True
		self.setIdCamera(0)
		print("thread init (video capture)")
		self.dirr = os.getcwd()
		self.imageDimension = [480, 640, 3]
		self.imagePixelNum = self.imageDimension[0]*self.imageDimension[1]*self.imageDimension[2]
		#self.BG = QtGui.QPixmap(self.dirr+"/images/BG.jpg")
		global product
		product=["",""]



	def setIdCamera(self, idCamera):
		self.idCamera = idCamera


	def run(self):
		global product, duration
		self.cap = cv2.VideoCapture(self.idCamera)
		w, h = self.video_size
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)  # ADAPT
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)  # ADAPT
		self.ThreadActive = True

		while self.ThreadActive:
			#print("thread run (video capture)"+ str(self.ThreadActive))
			ret, frame = self.cap.read()

			if ret:
				rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				rgbImage = cv2.flip(rgbImage, 1)
				#rgbImage = cv2.resize(rgbImage, (200,200), interpolation = cv2.INTER_AREA)
				convertToQtFormate = QtGui.QImage(rgbImage.data,rgbImage.shape[1], rgbImage.shape[0], QtGui.QImage.Format_RGB888)
				convertToQtFormate = QtGui.QPixmap.fromImage(convertToQtFormate)
				pixmap = QPixmap(convertToQtFormate)
				imgToShow  = pixmap.scaled(w, h, 1)
				###t = Timer("accumulate")
				###t.start()

				#ie = imageEnhancment()
				#type(imgToShow)
				#imgToShow = ie.QPixmapToArray(imgToShow)
				#imgArray = ie.deleteBackGround(imgArray)
				#if not self.isWhite(rgbImage):
				predictedType = Algorithm.predictType(self, rgbImage)
				product= [predictedType[0], predictedType[1]]
				print(product)
				#else:
				#product= ["No","No"]

				###duration = t.stop()

				#print(predictedType)
				#imgToShow = Algorithm.processCapturedImg(self,imgToShow,"hello")
				self.ImageUpdate.emit(imgToShow)
	def isWhite(self,img):
		return  0
		#np.count_nonzero((img == [250, 250, 250]).all(axis=2))
		#img = cv2.threshold(img,220,220, cv2.THRESH_BINARY)

		WHITE_MIN = np.array([220, 220, 220], np.uint8)
		WHITE_MAX = np.array([255, 255, 255], np.uint8)

		dst = cv2.inRange(img, WHITE_MIN, WHITE_MAX)
		no_white = cv2.countNonZero(dst)
		print('The number of white pixels is: ' + str(no_white) + "   "+str(self.imagePixelNum)+"       "+str(no_white*100/self.imagePixelNum)+" % " )
		return 0

	def getProduct(self):
		global product
		return product

	def getDuaration(self):
		global duration
		return duration

	def stop(self):
		self.cap.release()
		#self.out.release()
		self.ThreadActive = False
		#self.video_lbl.setPixmap(self.BG)

		self.quit()

	