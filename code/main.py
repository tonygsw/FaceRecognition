import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
from facedetect import detect_face
import facenet.nn4 as network
import random

import sklearn

from sklearn.externals import joblib

# face detection parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

# facenet embedding parameters

model_dir = './facenet/model_check_point/model-20170512-110547.ckpt-250000'
# 模型的路径
model_def = 'models.nn4'

image_size = 96
pool_type = 'MAX'  # "The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn = False  # "Enables Local Response Normalization after the first layers of the inception network."
seed = 42,  # "Random seed."
batch_size = None  # "Number of images to process in a batch."

frame_interval = 3  # frame intervals

def to_rgb(img):
	w, h = img.shape
	ret = np.empty((w, h, 3), dtype=np.uint8)
	ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
	return ret

print('Creating networks and loading parameters')

gpu_memory_fraction = 1.0
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

#restore facenet model
print('建立facenet embedding模型')
tf.reset_default_graph()
sess = tf.Session()

model_checkpoint_path='./facenet/model_check_point/model-20170512-110547.ckpt-250000'
saver = tf.train.import_meta_graph('./facenet/model_check_point/model-20170512-110547.meta')
saver.restore(sess, model_checkpoint_path)
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
print('facenet embedding模型建立完毕')

model = joblib.load('./facenet/model_check_point/knn_classifier.model')

video_capture = cv2.VideoCapture(0)
c = 0
while True:
	ret, frame = video_capture.read()

	timeF = frame_interval

	if (c % timeF == 0):#每3帧分析一次

		find_results = []
		img=frame
		bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

		nrof_faces = bounding_boxes.shape[0]  # number of faces
		# print('找到人脸数目为：{}'.format(nrof_faces))

		for face_position in bounding_boxes:

			face_position = face_position.astype(int)
			cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 2)

			crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
			if len(crop)<50:
				continue
			crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
			data = crop.reshape(-1, 160, 160, 3)

			emb_data = sess.run([embeddings],feed_dict={images_placeholder: np.array(data),phase_train_placeholder: False})[0]

			predict = model.predict(emb_data)

			if predict == 1:
				find_results.append('me')
				print("me")
			else:
				find_results.append('others')
				print("other")

		cv2.putText(frame, 'detected:{}'.format(find_results), (50, 100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),thickness=2, lineType=2)
	# print(faces)
	c += 1
	# Draw a rectangle around the faces

	# Display the resulting frame

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()