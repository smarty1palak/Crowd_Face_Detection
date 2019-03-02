from __future__ import print_function
import numpy as np
import cv2
import dlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

detector_hog = dlib.get_frontal_face_detector()

bgr_img = cv2.imread('./test.jpg', 1)
print (bgr_img.shape)

### detection
list_time = []
for idx in range(10):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    start = cv2.getTickCount()
    (h, w) = bgr_img.shape[:2]
    
    rgb_img = cv2.resize(rgb_img, None, fx=0.5, fy=0.5)
    dlib_rects = detector_hog(rgb_img, 1)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    list_time.append(time)
    # print ('elapsed time: %.3fms'%time)

print ('dlib hog average time: %.3f ms'%np.array(list_time[1:]).mean())

### draw rectangle bbox
c=0
if args.with_draw == 'True':
    for dlib_rect in dlib_rects:
        l = dlib_rect.left() * 2
        t = dlib_rect.top() * 2
        r = dlib_rect.right() * 2
        b = dlib_rect.bottom() * 2

        cv2.rectangle(bgr_img, (l,t), (r,b), (0,255,0), 2)
        c= c+1

    cv2.namedWindow('show', 0)
    cv2.imshow('show', bgr_img)
    print("no of faces detected: ", c)
    cv2.waitKey()