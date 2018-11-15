# FaceRecognition

	基于MTCNN和facenet的人脸识别实现

### tip

	author: gsw——HNU
	data: 2018
	tips: 这是pycharm项目,最好用pycharm打开
	模型链接放在了issues里面

### File Directory

	./code: 代码文件
	./code/20170512-110547: 模型备份处
	./code/dataprocess: 数据处理,包括图片处理，裁剪
	./code/facedetect: 人脸检测
	./code/facenet: 人脸识别
	./code/luzhi: 人脸录制
	./main :主函数
	./train_your_classifier: 模型训练

### how to use it

	(1)使用luzhi把需要检测的人脸通过摄像头录制进去
	(2)使用train_your_classifier训练模型,如果想要准确率更高，可以使用图片旋转，加噪等方法
	(3)使用main进行识别
	(4)友情说明:在一般的电脑上跑，比较慢，画面会比较卡顿
