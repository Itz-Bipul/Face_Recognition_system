FACE RECOGNITION ATTENDANCE SYSTEM
================================================

This project is a real-time attendance system based on face recognition using machine learning and computer vision.

**The system works in two main stages.

>>train.py
This script performs the training process. It reads face images captured from a live camera or stored in the dataset, extracts facial feature encodings, and saves them for future recognition.

>>live_attendance.py
This script handles real-time attendance. It uses a live webcam feed to detect and recognize faces by comparing them with the trained encodings. When a person is successfully recognized, the system captures the face image and records the name, date, and time in an Excel attendance file.


**HOW TO RUN THE FACE RECOGNITION ATTENDANCE SYSTEM
================================================

Follow these steps carefully to run the project on a new local machine.

------------------------------------------------
1. SYSTEM REQUIREMENTS
------------------------------------------------
- Python 3.10
- Webcam (built-in or external)
- Operating System: Windows / macOS / Linux

------------------------------------------------
2. INSTALL CONDA (RECOMMENDED)
------------------------------------------------
Download and install Miniforge (Conda):
https://github.com/conda-forge/miniforge

After installation, open a new terminal and verify:
conda --version

------------------------------------------------
3. CREATE AND ACTIVATE ENVIRONMENT
------------------------------------------------
Run the following commands in terminal:

conda create -n faceenv310 python=3.10 -y
conda activate faceenv310

------------------------------------------------
4. INSTALL REQUIRED LIBRARIES
------------------------------------------------
Run:
conda install -c conda-forge face_recognition opencv numpy openpyxl -y

Verify installation:
python -c "import cv2, face_recognition; print('Setup OK')"

------------------------------------------------
5. PROJECT SETUP
------------------------------------------------
Place all project files in one folder, for example:

FaceRecognition/
- Dataset/
- train.py
- live_attendance.py
-Attendance.xlsx

------------------------------------------------
6. ADD FACE DATA (TRAINING)
------------------------------------------------
You can add faces in two ways:
- Capture faces using live camera (recommended), may be 
- Place clear face images manually inside Dataset/

Rules:
- One face per image
- Image filename should match the person name
- Good lighting and front-facing images

------------------------------------------------
7. TRAIN THE SYSTEM
------------------------------------------------
Run training once (or after adding new faces):

python train.py

This will generate:
- encodings.npy
- names.npy

------------------------------------------------
8. START LIVE ATTENDANCE
------------------------------------------------
Run:
python live_attendance.py

What happens:
- Webcam opens
- Face is detected and recognized
- Face image is saved in CapturedFaces/
- Attendance is recorded in Attendance.xlsx

Press Q to close the camera.

------------------------------------------------
9. MAC CAMERA PERMISSION (macOS ONLY)
------------------------------------------------
If camera does not open:
System Settings -> Privacy & Security -> Camera
Enable Terminal / Python
Restart terminal and try again

------------------------------------------------
10. OUTPUT FILES
------------------------------------------------
- Attendance.xlsx : Attendance records
- CapturedFaces/  : Detected face images

------------------------------------------------
END
------------------------------------------------
