from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def main(args):
    
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    
    filename =args.input 
    output_filename =args.output


    draw = cv2.imread(filename)

    img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
    
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]

    #print("boundingbox: ",bounding_boxes);
    #print("points:",points)

    for b in bounding_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
        print(b)



    for p in points.T:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    cv2.imwrite(output_filename,draw)
                            
    print('Total %d face(s) detected, saved in %s' % (nrof_faces,output_filename))
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image to be detected for faces.',default='./test.jpg')
    parser.add_argument('--output', type=str, help='new image with boxed faces',default='new.jpg')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
