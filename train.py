import cv2
import face_recognition
import os
import numpy as np

# -------- DATASET PATH --------
dataset_path = "/Users/bipuldas/Downloads/Face_Recognition_System/Dataset"
print("Dataset files:", os.listdir(dataset_path))

images = []
classNames = []

for file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, file))
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(file)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)

np.save("encodings.npy", encodeListKnown)
np.save("names.npy", classNames)

print("✅ Training complete")
print("Total faces trained:", len(classNames))