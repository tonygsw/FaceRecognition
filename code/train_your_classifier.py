import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
from facedetect import detect_face
from facenet import nn4 as network
from facenet import facenet
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='./facenet/model_check_point/model-20170512-110547.ckpt-250000'
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96
pool_type='MAX'
use_lrn=False
seed=42
batch_size= None

#建立人脸检测模型，加载参数

print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,None)
#建立facenet embedding模型
print('建立facenet embedding模型')
tf.reset_default_graph()
sess = tf.Session()
#images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')

#phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

#embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)

#ema = tf.train.ExponentialMovingAverage(1.0)
#saver = tf.train.Saver(ema.variables_to_restore())

model_checkpoint_path='./facenet/model_check_point/model-20170512-110547.ckpt-250000'
saver = tf.train.import_meta_graph('./facenet/model_check_point/model-20170512-110547.meta')

saver.restore(sess, model_checkpoint_path)
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
print('facenet embedding模型建立完毕')

###### train_dir containing one subdirectory per image class 
#should like this:
#-->train_dir:
#     --->pic_me:
#            me1.jpg
#            me2.jpg
#            ...
#     --->pic_others:
#           other1.jpg
#            other2.jpg
#            ...
data_dir='./dataporcess/picture/'#your own train folder

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        data[guy] = curr_pics
    return data
data=load_data(data_dir)
keys=[]
for key in data.keys():
    keys.append(key)
    print('foler:{},image numbers：{}'.format(key,len(data[key])))
train_x=[]
train_y=[]

for x in data[keys[0]]:
    crop = cv2.resize(x, (160, 160), interpolation=cv2.INTER_CUBIC)

    crop_data=crop.reshape(-1,160,160,3)

    crop_data=crop_data.astype(int)
    #print(crop_data.shape)

    emb_data =  list(sess.run([embeddings], feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False })[0])

    train_x.append(emb_data)
    train_y.append(0)
print(len(train_x))


for y in data[keys[1]]:
    crop = cv2.resize(y, (160, 160), interpolation=cv2.INTER_CUBIC )

    crop_data=crop.reshape(-1,160,160,3)
    #print(crop_data.shape)
    emb_data = list(sess.run([embeddings], feed_dict={images_placeholder: np.array(crop_data), phase_train_placeholder: False })[0])

    train_x.append(emb_data)
    train_y.append(1)
    

print(len(train_x))
print('搞完了，样本数为：{}'.format(len(train_x)))
#train/test split
train_x=np.array(train_x)
train_x=train_x.reshape(-1,128)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  

classifiers = knn_classifier 

model = classifiers(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy))

#save model
joblib.dump(model, './facenet/model_check_point/knn_classifier.model')
#model = joblib.load('_2017_1_24_knn.model')
model = joblib.load('./facenet/model_check_point/knn_classifier.model')
predict = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy))