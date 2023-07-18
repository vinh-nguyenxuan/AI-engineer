import cv2
import os
import numpy as np
import pickle

image_dir = os.path.join("datasets", "images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    # print(root, dirs, files)
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-")
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                id_ = label_ids[label]
                # print(path)
            pil_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
            # final_image = pil_image.resize(size, Image.ANTIALIAS)
            # image_array = np.array(pil_image, "uint8")
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = gray[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            # images = cv2.imread(path)
    
            # print(faces)


with open("face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")