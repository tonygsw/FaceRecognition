import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from dataporcess.get_picture import load_data, resize_image, IMAGE_SIZE


class Data:
	def __init__(self, path_name, ClassNum = 2):
		#训练集
		self.train_images = None
		self.train_labels = None
		#验证集
		self.valid_images = None
		self.valid_labels = None
		#测试集
		self.test_images = None
		self.test_labels = None

		#加载路径
		self.path_name=path_name

		#标签种类数
		self.ClassNum=ClassNum

	# 把标签变成[0,1]，[1,0]
	def processlables(self,ClassNum, labels):
		ans = []
		for lab in labels:
			a = [0] * int(ClassNum)
			a[lab] = 1
			ans.append(a)
		return ans
	#加载数据
	def load(self):
		images,labels=load_data(self.path_name)
		train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,random_state=random.randint(0, 100))

		_, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,random_state=random.randint(0, 100))
		# 输出训练集、验证集、测试集的数量
		print('train samples', train_images.shape[0])
		print('valid samples', valid_images.shape[0])
		print('test samples', test_images.shape[0])

		train_labels = self.processlables(self.ClassNum,train_labels)
		valid_labels = self.processlables(self.ClassNum,valid_labels)
		test_labels = self.processlables(self.ClassNum,test_labels)

		train_images = train_images.astype('float32')
		valid_images = valid_images.astype('float32')
		test_images = test_images.astype('float32')

		train_images /= 255
		valid_images /= 255
		test_images /= 255
		self.train_images = train_images
		self.valid_images = valid_images
		self.test_images = test_images
		self.train_labels = train_labels
		self.valid_labels = valid_labels
		self.test_labels = test_labels
	def next_batch(self,data,label,size=10):
		datas=[]
		labs=[]
		for i in range(size):
			a=int(random.random()*len(data[0]))
			datas.append(data[a])
			labs.append(label[a])
		return datas,labs

if __name__=='__main__':
	dataset = Data("E:\\university\\CNN\TensorFlow\\facetest\\get_picture\\picture")
	dataset.load()


