import cv2
import numpy as np
import imutils

# Kernels
sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
blur_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
ledge_kernel = np.array([[-1,1,0],[-1,1,0],[-1,1,0]])
emboss_kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
sobel_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
motion_kernel = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
cap = cv2.VideoCapture('Mission_Impossible.mp4')
ret, img1 = cap.read()
img1 = imutils.resize(img1, width=600)

while(1):
	ret, img1 = cap.read()
	image = imutils.resize(img1, width=600)
	cv2.imshow('img1', image)

	img = cv2.filter2D(image, -1, edge_kernel)
	img_new = 1 - img
	cv2.imshow('Convolved', img)
	cv2.imshow('negative', img_new)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break