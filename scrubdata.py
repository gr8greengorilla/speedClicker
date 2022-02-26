

from turtle import pd
import pandas as pd
import mediapipe
import cv2
import os

path = r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\datalabels.csv"
data = pd.read_csv(path)
i = 0

mpPose = mediapipe.solutions.pose
pose = mpPose.Pose()

todrop = []
while True:
    try:
        fname = data.iloc[i, 0]
        print(fname)
    except (IndexError):
        break

    try:
        img = cv2.imread(fname)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        results = pose.process(imgRGB).pose_landmarks.landmark
        landmarks = [results[15],results[16]] + [results[x] for x in range(23,32)]
        _ = [i.x for i in landmarks] + [i.y for i in landmarks] + [i.z for i in landmarks] + [i.visibility for i in landmarks]
    except:
        print(f"Dropping {fname}")
        todrop.append(i)
        try:
            os.remove(fname)
        except:
            pass
    
    i += 1

data = data.drop(todrop)

data.to_csv("datalabels2.csv", index=False)
print("Done")