import os
import sys
import cv2
import numpy as np

IMAGE_SIZE=64


#调整图片大小

def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
	top, bottom, left, right=(0, 0, 0, 0)

	h, w, _ = image.shape
	longest_edge =max(h, w)
	if h<longest_edge:
		dw = longest_edge-h
		top = dw // 2
		bottom = dw //2
	elif w<longest_edge:
		dw = longest_edge -w
		left = dw //2
		right =dw -left
	else:
		pass
	#加黑边
	constant = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))

	return cv2.resize(constant,(height, width));

images = []

labels = []


def read_path(path_name):
	for dir_item in os.listdir(path_name):
		full_path = os.path.abspath(os.path.join(path_name,dir_item))

		if os.path.isdir(full_path):
			read_path(full_path)
		else:
			if dir_item.endswith('.jpg'):
				print(full_path)
				image=cv2.imread(full_path)

				image=resize_image(image)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

				#cv2.imwrite("./j.jpg",image)
				images.append(image)
				labels.append(path_name)
	return images,labels


def load_data(path_name):
	images, labels=read_path(path_name)

	images = np.array(images)
	#print(images.shape)

	labels=np.array([0 if label.endswith('me') else 1 for label in labels])

	return images,labels


if __name__=='__main__':
	images,labels=load_data("C:\\batch16")