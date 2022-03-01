from tkinter import E
import cv2
from trainnet import Net
import torch.nn as nn
import torch
import mediapipe



model = Net()
model.load_state_dict(torch.load(r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\saved_models\15-1ke5tv.pt", map_location=torch.device("cpu")))
model.eval()

mpPose = mediapipe.solutions.pose
pose = mpPose.Pose()

def getPose(img):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB).pose_landmarks.landmark

    landmarks = [results[15],results[16],results[30],results[31]] + [results[x] for x in range(23,26)]

    inputs = [x.x for x in landmarks] + [x.y for x in landmarks] + [x.z for x in landmarks]

    return inputs

fnames = ["test" + str(i) for i in range(1,8)]
for i in fnames:
    vidcap = cv2.VideoCapture(f'tests/{i}.mp4')
    success,image = vidcap.read()
    numfailed = 0
    print(f"Processing {i}...", end="\r")
    inputs = []
    tryagain = 0
    while success:
        if tryagain < 3:
            try:
                inputs.append(getPose(image))
                tryagain = 0
                success,image = vidcap.read()
            except:
                tryagain += 1
        else:
            numfailed += 1
            tryagain = 0
            success,image = vidcap.read()
        
        

    inputs = torch.Tensor(inputs)
    output = []
    for x in inputs:
        output.append(model(x).argmax(0).item())

    output2 = []
    for j in range(len(output)-1):
        if output[j] != output[j+1]:
            output2.append(output[j])

    total = sum(output2)
    print(f"{i}: {total:>3d} with {numfailed:>2d} missed frames")
