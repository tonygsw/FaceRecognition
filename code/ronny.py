
'''
备份
'''

import cv2
import sys
from  PIL import Image
import tensorflow as tf
from facedetect import detect_face
from facenet.network import Model
#from luzhi import RecordingFace



def Catchvideo(WindowName, CameraId):
	CNN = Model(64 * 64 * 1, 2)

	cv2.namedWindow(WindowName)
	cap=cv2.VideoCapture(CameraId)

	sess = tf.Session()
	pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

	minsize = 40  # minimum size of face
	threshold = [0.6, 0.7, 0.9]  # three steps's threshold
	factor = 0.709  # scale factor

	while cap.isOpened():
		ok,frame=cap.read()
		if not ok:
			break
		#灰度转换
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
		nrof_faces = bounding_boxes.shape[0]
		for b in bounding_boxes:
			img = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
			resule=CNN.mode(False,img)
			resule=resule[0]
			if(resule[0]>resule[1]):
				cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
				cv2.putText(frame, 'me', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 225), 4)
		cv2.imshow(WindowName, frame)
		if cv2.waitKey(10) & 0xff==ord('q'):
			break


if __name__=='__main__':
	Catchvideo("me",0)
