from operator import mod
import cv2
import torch
import mediapipe
import numpy as np
from trainnet import Net

class Clicker():
    def __init__(self, model=None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not model:
            self.model = Net()
            self.model.load_state_dict(torch.load(r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\saved_models\MostRecent.pt", map_location=torch.device(self.device)))
            self.model.eval()
        else:
            self.model = model

        self.mpPose = mediapipe.solutions.pose
        

    def getPose(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(imgRGB).pose_landmarks.landmark

        inputs = [[x.x,x.y,x.z] for x in results]

        return inputs
    
    def setModel(self, model):
        self.model = model

    def getScore(self, vdata, n=1):
        vdata = vdata[:,[15,16,23,24,25,26,31,32],:] #900 Maxes the amount of frames per video, [15,16...] selects the landmarks wanted to use
        totaldata = []
        for j in range(n, len(vdata)-n):
            slice = vdata[j-n:j+n+1]     
            totaldata.append(slice)

        totaldata = np.array(totaldata)
        totaldata = totaldata.reshape([totaldata.shape[0],-1])

        with torch.no_grad():
            self.model = self.model.to(self.device)
            with torch.no_grad():
                outputs = [self.model(torch.tensor(x).to(self.device)).argmax().item() for x in totaldata]

        clicks = []
        for i in range(len(outputs)-1):
            if outputs[i] != outputs[i+1]:
                clicks.append(outputs[i])
        if outputs[-1] != outputs[-2]:
            clicks.append(outputs[-1])

        score = clicks.count(1)

        return score
    
    def labelVideo(self, path):
        self.pose = self.mpPose.Pose()

        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()

        vdata = []

        count = 0
        while success:
            print(f"Frame {count:4d}", end="\r")
            count += 1
            try:
                vdata.append(self.getPose(image))
            except:
                print(f"Failed to create landmarks on frame {count:4d}")
            success,image = vidcap.read()

                

        vdata = np.array(vdata)
        return vdata


def main():
    thing = Clicker()

    scores = []
    for i in range(1,9):
        data = np.load(f"tests/test{i}.npy")
        scores.append(thing.getScore(data))
    print(scores)

if __name__ == "__main__":
    main()
