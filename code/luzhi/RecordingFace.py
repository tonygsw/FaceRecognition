import sys
import cv2
import tensorflow as tf
from facedetect import detect_face
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def Catchvideo(Windowname, cameraId, NumPic, Path):
	cv2.namedWindow(Windowname)
	cap=cv2.VideoCapture(cameraId)

	sess = tf.Session()
	pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
	num = 0
	minsize = 40  # minimum size of face
	threshold = [0.6, 0.7, 0.9]  # three steps's threshold
	factor = 0.709  # scale factor

	while cap.isOpened():
		ok,frame=cap.read()
		if not ok:
			break
		bounding_boxes, points = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
		nrof_faces = bounding_boxes.shape[0]
		if(len(bounding_boxes)>1):
			continue

		for b in bounding_boxes:
			img = frame[int(b[1]):int(b[3]),int(b[0]):int(b[2])]
			if(len(img)<60):
				continue
			imgname = Path + "/" + "%d.jpg" % (num)
			num += 1
			print(imgname)

			cv2.imwrite(imgname, img)
			cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
		cv2.imshow(Windowname, frame)
		if num > NumPic:
				break
		if cv2.waitKey(10) & 0xff==ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	path="../dataporcess/picture/me"
	Catchvideo("me",0,1000,path)