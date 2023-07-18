import numpy as np
import cv2
import pickle
# import attendance
from datetime import datetime
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels = {"person_name": 1}
with open("face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def round_int(x):
    if x in [float("-inf"),float("inf")]: return float("nan")
    return int(round(x))

cap = cv2.VideoCapture(0)

while(True):
    
    ret, frame = cap.read()
    start = time.perf_counter()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        
        if conf>=30 and conf<=75:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (101, 201, 111)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            print(name)
            # markAttendance(name)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Unknown"
            color = (101, 201, 111)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0) 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        markAttendance(name)
    
    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime 
    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('\x1b'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()