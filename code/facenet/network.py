import math
import os
import tensorflow as tf
import cv2
from dataporcess.porcess import Data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resize_image(image, height=64, width=64):
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

	constant=cv2.resize(constant,(height, width))
	image = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
	image = image.astype('float32')
	return image/255


class CNNModel:
	def prelu(self, inp, name):
		with tf.variable_scope(name):
			i = int(inp.get_shape()[-1])
			alpha = self.make_var('alpha', shape=(i,))
			output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
		return output

	def weight_variable(self, shape, name):
		initial = tf.truncated_normal(shape, stddev=0.1)  # 从截断的正态分布中输出随机值。
		return tf.Variable(initial, name=name)

	def bias_variable(self, shape, names):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name=names)

	def make_var(self, name, shape):
		'''Creates a new TensorFlow variable.'''
		return tf.get_variable(name, shape)

	# 卷积(845, 64, 64, 1)
	def conv2d(self, x, W, strides, padding, name):
		return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)

	# 池化
	def max_pool(self, x, ksize, strides, padding, name):
		return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding,name=name)

class Model(CNNModel):
	def __init__(self,input,output):
		self.input=input
		self.output=output
	def mode(self,TRAIN,date=None):
		tf.reset_default_graph()
		x = tf.placeholder("float", shape=[None, 64,64],name="input")
		y_ = tf.placeholder("float", shape=[None, self.output],name="output")
		sess = tf.InteractiveSession()

		x_image = tf.reshape(x, [-1,64,64,1])

		#输入层权重
		W_conv1 = self.weight_variable([3, 3, 1, 64], "W_conv1")
		b_conv1 = self.bias_variable([64], "b_conv1")
		#NUM 1层
		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1,[1, 1, 1, 1], padding='SAME',name="h_conv1") + b_conv1)
		h_pool1 = self.max_pool(h_conv1,[1, 2, 2, 1],[1, 2, 2, 1],"SAME","h_pool1")#输入层输出
		# 隐含层权重

		W_conv2 = self.weight_variable([3, 3, 64, 32],"W_conv2")
		b_conv2 = self.bias_variable([32],"b_conv2")
		# NUM 2层
		h_conv2 = self.prelu(self.conv2d(h_pool1, W_conv2,[1, 1, 1, 1], padding='SAME',name="h_conv2") + b_conv2,name="prule1")
		h_pool2 = self.max_pool(h_conv2,[1, 2, 2, 1],[1, 2, 2, 1], "SAME", "h_pool2")  # 隐含层输出
		#drop层
		keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_pool2, keep_prob)


		W_conv3 = self.weight_variable([3, 3, 32, 64], "W_conv3")
		b_conv3 = self.bias_variable([64], "b_conv3")
		#NUM 1层
		h_conv3 = self.prelu(self.conv2d(h_fc1_drop, W_conv3,[1, 1, 1, 1], padding='SAME',name="h_conv3") + b_conv3,name="prule2")
		h_pool3 = self.max_pool(h_conv3,[1, 2, 2, 1],[1, 2, 2, 1], "SAME", "h_pool3")#输入层输出
		# 隐含层权重
		W_conv4 = self.weight_variable([3, 3, 64, 64],"W_conv4")
		b_conv4 = self.bias_variable([64],"b_conv4")
		# NUM 2层
		h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4,[1, 1, 1, 1], padding='SAME',name="h_conv4") + b_conv4)
		h_pool4 = self.max_pool(h_conv4,[1, 2, 2, 1],[1, 2, 2, 1], "SAME", "h_pool4")  # 隐含层输出
		#drop层
		keep_prob1 = tf.placeholder("float")
		h_fc1_drop2 = tf.nn.dropout(h_pool4, keep_prob1)


		w_conv5 = self.weight_variable([16*64,1024],"h_conv5")
		b_conv5 = self.bias_variable([1024],"b_conv5")
		h_fc1_drop2=tf.reshape(h_fc1_drop2,[-1,1024])
		h_conv5= tf.nn.relu(tf.matmul(h_fc1_drop2, w_conv5) + b_conv5)

		W_conv6 = self.weight_variable([1024, 2],"W_conv6")
		b_conv6 = self.bias_variable([2],"W_conv6")

		y_conv=tf.nn.softmax(tf.matmul(h_conv5, W_conv6) + b_conv6)

		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_conv - y_),reduction_indices=[1]))
			tf.summary.scalar('loss', loss)

		with tf.name_scope('train'):
			train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		saver = tf.train.Saver(max_to_keep=1)

		if TRAIN:


			#sess.run(tf.global_variables_initializer())

			saver.restore(sess, "save/model.ckpt")

			dataset = Data("../dataporcess/picture")
			dataset.load()

			summary_op = tf.summary.merge_all()
			writer = tf.summary.FileWriter("logs/", sess.graph)

			for i in range(1000):
				date,lab=dataset.next_batch(dataset.train_images,dataset.train_labels)

				train_step.run(feed_dict={x: date, y_: lab, keep_prob: 0.75, keep_prob1: 0.75})

				if i%5==0:
					date, lab = dataset.next_batch(dataset.test_images, dataset.test_labels)
					result = sess.run(summary_op,feed_dict={x: date, y_: lab, keep_prob:1, keep_prob1:1})
					writer.add_summary(result, i)
					train_accuracy = accuracy.eval(feed_dict={x:date, y_: lab, keep_prob:1.0,keep_prob1:1.0})
					print("次数: %d,准确率: %g" % (i, train_accuracy))
			saver.save(sess, 'save/model.ckpt')
		else:
			saver.restore(sess, "save/model.ckpt")
			date=resize_image(date)
			y = sess.run(y_conv,feed_dict={x:[date],keep_prob:1.0,keep_prob1:1.0})
			return y


if __name__=='__main__':

	CNN=Model(64*64*1,2)
	a=cv2.imread("E:\\university\\CNN\\MTCNN\\dataporcess\\picture\\other\\10d.jpg")
	#a=[a]
	print(a.shape)
	ff=CNN.mode(False,a)
	print(ff)
