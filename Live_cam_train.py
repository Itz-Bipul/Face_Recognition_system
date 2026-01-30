import cv2
import face_recognition
import os
import numpy as np

# -------- CONFIG --------
DATASET_DIR = "Dataset"
NUM_IMAGES = 10   # number of face samples to capture

os.makedirs(DATASET_DIR, exist_ok=True)

person_name = input("Enter person name: ").strip().lower()

cap = cv2.VideoCapture(0)
count = 0

print("📸 Camera started")
print("Press SPACE to capture face")
print("Press Q to quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)

    for (top, right, bottom, left) in faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Images captured: {count}/{NUM_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Live Training", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE to capture
        if len(faces) == 1:
            img_path = f"{DATASET_DIR}/{person_name}_{count}.jpg"
            cv2.imwrite(img_path, frame)
            count += 1
            print(f"✅ Saved {img_path}")
        else:
            print("⚠️ Ensure exactly ONE face is visible")

    if key == ord('q') or count >= NUM_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print("📦 Image capture completed")