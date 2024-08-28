from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

model = YOLO("best.pt")
cap = cv2.VideoCapture("../data/1.mp4") #for video


#model


classNames = ["child","adult"]

tracker = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img,stream = True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding
            #opencv
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # 2550255 is color 3 is thickness

            #cvzone
            w,h = x2 -x1 , y2 -y1


            #print(x1, y1, x2, y2)

            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            # classname
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == "child" or currentClass == "adult")  and conf>0.3 :

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt = 5)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections  = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt = 2, colorR= (255,0,0) )
        cvzone.putTextRect(img, f'ID:{id}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

    #blue one tracker detecting purple is yolo detecting