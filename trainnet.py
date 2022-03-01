import torch
import torch.nn as nn
import torchvision
import torch.utils.data as tud
import pandas as pd
from PIL import Image
import cv2
import mediapipe
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 1000

class SpeedDataset(tud.Dataset):

    def __init__(self, path, pddata, transform=None, train=True, split=.20):
        self.path = path + "/"
        self.transform = transform
        self.data = pddata
        #self.data = self.data.sample(frac = 1)
        if train:
            self.data = self.data.iloc[int(len(self.data) * split):]
        else:
            self.data = self.data.iloc[:int(len(self.data) * split)]

        self.mpPose = mediapipe.solutions.pose
        self.pose = self.mpPose.Pose()

        self.poses = []
        todrop = []
        for i in range(0,len(self.data)):
            print(f"Processing frames: {i:>5d}/{len(self.data):>5d}", end="\r")
            fname = self.path + self.data.iloc[i, 0]
            trylimit = 0
            while True:
                try:
                    landmarks = self.getPose(fname)
                    input = [x.x for x in landmarks] + [x.y for x in landmarks] + [x.z for x in landmarks]

                    self.poses.append(input)
                    break
                except:
                    trylimit += 1
                    if trylimit == 5:
                        print(f"{i} Failed to make landmarks for " + fname)
                        todrop.append(i)
                        break
                        

        if todrop:
            todrop.reverse()
            self.data.drop([self.data.index[x] for x in todrop], inplace=True)
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        category = self.data.iloc[idx, 1]
        sample = (torch.Tensor(self.poses[idx]), torch.tensor(category, dtype=torch.long))

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample
    
    def getPose(self, pathToImg):
        img = cv2.imread(pathToImg)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB).pose_landmarks.landmark

        landmarks = [results[15],results[16],results[30],results[31]] + [results[x] for x in range(23,26)]

        return landmarks
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(21, 15), #Did 12 12 12 12, trying 17 17 12 12 
            nn.ReLU(),
            #nn.Linear(12,12),
            #nn.ReLU(),
            nn.Linear(15,2)
        )
        
    
    def forward(self, x):
        x = self.network(x)
        return x


def trainloop(dataloader, model, optimizer, loss_fn, verbose=True):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 and verbose:
            print(f"Loss: {loss.item():>7f}, [{batch * len(X):>5d}/{size:>5d}]")

def testloop(dataloader, model, loss_fn, verbose=True):
    totalloss, ncorrect = 0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        totalloss += loss_fn(pred, y).item()
        ncorrect += (y == pred.argmax(1)).type(torch.float).sum().item()
    
    if verbose:
        ncorrect /= size
        totalloss /= num_batches
        print(f"Accuracy: {ncorrect * 100:>2f}%, Avg loss = {totalloss:5f}")

def main():

    path = "C:/Users/erik/OneDrive/Cross-Code/VSCode/PythonCode/Starting_Code/Machine Learning/pytorchlearning/Speed"
    pddata = pd.read_csv("datalabels.csv").sample(frac=1)
    testdata = SpeedDataset(path, pddata, train=False)
    traindata = SpeedDataset(path, pddata, train=True)
    

    trainloader = tud.DataLoader(traindata, batch_size=8, shuffle=True)
    testloader = tud.DataLoader(testdata, batch_size=8, shuffle=True)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_epochs):
        verbose = i % 10 == 0
        if verbose:
            print(f"Epoch {i:>3d}/{num_epochs:>3d}---------------")
        trainloop(trainloader, model, optimizer, loss_fn, False)
        testloop(testloader, model, loss_fn, verbose)
    torch.save(model.state_dict(), "15-1ke5tv.pt")



if __name__ == "__main__":
    main()

