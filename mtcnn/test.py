
from __future__ import print_function
from mtcnn import detect_faces, show_bboxes
import numpy as np
import cv2
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()
with open('label.json') as f:
    label = json.load(f)
d=1
w=0
total =0
count=0
filename = "test"+ str(d) + ".jpg"
folder= "test_images"
for filename in os.listdir(folder):
	print( filename)
	bgr_img = cv2.imread(os.path.join(folder,filename))
	print (bgr_img.shape)
	d= d+1
	c1= label[filename]
	filename= "test"+ str(d) + ".jpg"
	### detection
	list_time = []
	for idx in range(10):
	    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	    start = cv2.getTickCount()
	    bounding_boxes, landmarks = detect_faces(rgb_img)
	    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
	    list_time.append(time)

	print ('mtcnn average time: %.3f ms'%np.array(list_time[1:]).mean())

	### draw rectangle bbox
	c=0
	if args.with_draw == 'True':
	    for b in bounding_boxes:
	        b = [int(round(value)) for value in b]
	        cv2.rectangle(bgr_img, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
	        crop_img=bgr_img[b[1]:b[3],b[0]:b[2]]
	        #cv2.imwrite("face-" + str(d-1) + str(c) + ".jpg", crop_img)
	        path = "cropped"
	        cv2.imwrite(os.path.join(path , "image-" +str(d-1)+ "face-" +  str(c) + ".jpg"), crop_img)
	        c= c+1
	        
	    for p in landmarks:
	        for i in range(5):
	            cv2.circle(bgr_img, (p[i] , p[i + 5]), 3, (255,0,0), -1)
	    print("no of faces detected: ", c)
	    w= w+ (c/c1)
	    total = total + c1
	    count= count+c
final = abs(count-total)/total
print("accuracy percentage: ", 100-final*100)
