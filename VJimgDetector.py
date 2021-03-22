import numpy as np
import os.path
import cv2
from readTxt import getdf, bb_intersection_over_union, IoU


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

df = getdf()
for i in range(162,320):
    if os.path.isfile('data-faces-wider/' + str(i) + '.jpg'):

        img = cv2.imread('data-faces-wider/' + str(i) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.05, 4)

        if len(faces) > 0 :
            act_df = df[df['name'] == str(i) + '.jpg']
            tp=0
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                for index, row in act_df.iterrows():
                    xa = int(row['x'])
                    ya = int(row['y'])
                    wa = int(row['w'])
                    ha = int(row['h'])
                    iou = IoU(np.array([x, y, x + w, y + h]), np.array([xa, ya, xa + wa, ya + ha]))
                    print(iou)
                    if iou > 0.5:
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                        tp+=1

            fp= faces.shape[0] -tp
            fn=len(act_df) - tp

            for index, row in act_df.iterrows():
                x = int(row['x'])
                y = int(row['y'])
                w = int(row['w'])
                h = int(row['h'])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)


            print("true positives :" + str(tp))
            print("false positives :" + str(fp))
            print("false negatives :" + str(fn))

            cv2.imwrite('C:/Users/nagyb/PycharmProjects/Biom2/faces/' + str(i) + '.jpg',img)

            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
