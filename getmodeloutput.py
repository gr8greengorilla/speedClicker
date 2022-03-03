import cv2
from trainnet import Net
import torch
import mediapipe



class Clicker():
    def __init__(self) -> None:
        self.model = Net()
        self.model.load_state_dict(torch.load(r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\saved_models\15-1ke5tv.pt", map_location=torch.device("cpu")))
        self.model.eval()

        self.mpPose = mediapipe.solutions.pose
        

    def getPose(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(imgRGB).pose_landmarks.landmark

        landmarks = [results[15],results[16],results[30],results[31]] + [results[x] for x in range(23,26)]

        inputs = [x.x for x in landmarks] + [x.y for x in landmarks] + [x.z for x in landmarks]

        return inputs

    def getScore(self, frames):
        print("Processing new set of frames...")
        self.pose = self.mpPose.Pose()
        inputs = []
        count = 0
        for i in frames:
            print(f"Frame {count}...", end="\r")
            try:
                inputs.append(self.getPose(i))
            except:
                print(f"Failed to make landmarks for frame {count}")
            
            count += 1

        inputs = torch.Tensor(inputs)
        clicks = [self.model(x).argmax(0).item() for x in inputs]

        output = []
        for i in range(len(clicks)-1):
            if clicks[i] != clicks[i+1]:
                output.append(clicks[i])
        if clicks[-1] != clicks[-2]:
            output.append(clicks[-1])
        sum = 0
        for i in output:
            if i == 1:
                sum += 1
        
        print(f"Scored {sum}            ")
        return sum