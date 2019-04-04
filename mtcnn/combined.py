from __future__ import print_function
import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
import sys
from importlib import reload
import io
import gzip
from mtcnn import detect_faces, show_bboxes
import json
# we are only going to use 4 attributes
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
parser.add_argument('--img_dir', type=str,
                        default='cropped/', required = True,
                        help='input image directory (default: cropped/)')
parser.add_argument('--output_dir', type=str,
                        default='results/',
                        help='output directory to save the results (default: results/')
parser.add_argument('--model', type=str,
                        default='face_model.pkl', required = True,
                        help='path to trained model (default: face_model.pkl)')
def perform_mtcnn_shits():
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
        path1= "bounding_box"
        #cv2.imwrite("face-" +  str(d-1) + ".jpg", bgr_img)
    final = abs(count-total)/total
    print("accuracy percentage: ", 100-final*100)

def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs

def predict_one_image(img_path, clf, labels):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings),
                        columns = labels)
    pred = pred.loc[:, COLS]
    return pred, locs
def draw_attributes(img_path, df):
    """Write bounding boxes and predicted face attributes on the image
    """
    img = cv2.imread(img_path)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'

        race = np.argmax(row[1][1:4])
        text_showed = "{} {}".format(race, gender)
        new = cv2.resize(img, (0,0), fx=3, fy=3) 
        #cv2.rectangle(new, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = new.shape[1]
        cv2.putText(new, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    return new



def main():
    perform_mtcnn_shits()
    args = parser.parse_args()
    output_dir = args.output_dir
    input_dir = args.img_dir
    model_path = args.model

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # load the model
    with io.open(model_path,'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        #p = u.load()
        clf, labels = u.load()

    print("classifying images in {}".format(input_dir))
    for fname in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, fname)
        try:
            pred, locs = predict_one_image(img_path, clf, labels)
        except:
            print("Skipping {}".format(img_path))
            continue
        if not locs:
            continue
        locs = \
            pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
        df = pd.concat([pred, locs], axis=1)
        img = draw_attributes(img_path, df)
        cv2.imwrite(os.path.join(output_dir, fname), img)
        os.path.splitext(fname)[0]
        output_csvpath = os.path.join(output_dir,
                                      os.path.splitext(fname)[0] + '.csv')
        df.to_csv(output_csvpath, index = False)

if __name__ == "__main__":
    main()
