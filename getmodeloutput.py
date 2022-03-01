import cv2
from trainnet import Net
import torch.nn as nn
import torch
import mediapipe

vidcap = cv2.VideoCapture('tests/test5.mp4')
success,image = vidcap.read()
count = 0

model = Net()
model.load_state_dict(torch.load("saved_models/10-10-1ke.pt", map_location=torch.device("cpu")))
model.eval()

mpPose = mediapipe.solutions.pose
pose = mpPose.Pose()

def getPose(img):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB).pose_landmarks.landmark

    landmarks = [results[15],results[16],results[30],results[31]] + [results[x] for x in range(23,26)]

    inputs = [x.x for x in landmarks] + [x.y for x in landmarks] + [x.z for x in landmarks]

    return inputs


inputs = []
count = 0
while success:   
    print(f"processing frame {count}", end="\r")
    try:
        inputs.append(getPose(image))
    except:
        print(f"Failed to make landmarks for frame {count}")
    
    count += 1
    success,image = vidcap.read()

inputs = torch.Tensor(inputs)
output = []
for x in inputs:
    output.append(model(x).argmax(0).item())
print(output)
output2 = []
for i in range(len(output)-1):
    if output[i] != output[i+1]:
        output2.append(output[i])
sum = 0
for i in output2:
    if i == 1:
        sum += 1
print(sum)