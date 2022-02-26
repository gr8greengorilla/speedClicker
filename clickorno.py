from msilib import datasizemask
from numpy import corrcoef
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as tud
import pandas as pd
from PIL import Image
import cv2
import mediapipe

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 1000

class SpeedDataset(tud.Dataset):

    def __init__(self, path, csv_file, transform=None, train=True, split=.20):
        self.path = path + "/"
        self.transform = transform

        self.data = pd.read_csv(self.path + csv_file)
        if train:
            self.data = self.data.iloc[int(len(self.data) * split):]
        else:
            self.data = self.data.iloc[:int(len(self.data) * split)]

        
        self.mpPose = mediapipe.solutions.pose
        self.pose = self.mpPose.Pose()
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.path + self.data.iloc[idx, 0]

        landmarks = self.getPose(fname)

        input = [x.x for x in landmarks] + [x.y for x in landmarks] + [x.z for x in landmarks]# + [x.visibility for x in landmarks]

        category = self.data.iloc[idx, 1]
        sample = (torch.Tensor(input), torch.tensor(category, dtype=torch.long))

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample

    def getPose(self, pathToImg):
        img = cv2.imread(pathToImg)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            results = self.pose.process(imgRGB).pose_landmarks.landmark
        except (AttributeError):
            print(f"Cant identify {pathToImg}")

        landmarks = [results[15],results[16],results[30],results[31]] + [results[x] for x in range(23,26)]
        return landmarks
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(21, 12),
            nn.ReLU(),
            nn.Linear(12,12),
            nn.ReLU(),
            nn.Linear(12,2)
        )
        
    
    def forward(self, x):
        x = self.network(x)
        return x


def trainloop(dataloader, model, optimizer, loss_fn):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"Loss: {loss.item():>7f}, [{batch * len(X):>5d}/{size:>5d}]")

def testloop(dataloader, model, loss_fn):
    totalloss, ncorrect = 0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        totalloss += loss_fn(pred, y).item()
        ncorrect += (y == pred.argmax(1)).type(torch.float).sum().item()
    
    ncorrect /= size
    totalloss /= num_batches
    print(f"Accuracy: {ncorrect * 100:>2f}%, Avg loss = {totalloss:5f}")

        

def main():

    path = r"C:/Users/erik/OneDrive/Cross-Code/VSCode/PythonCode/Starting_Code/Machine Learning/pytorchlearning/Speed"
    traindata = SpeedDataset(path, "datalabels.csv", train=True)
    testdata = SpeedDataset(path, "datalabels.csv", train=False)

    trainloader = tud.DataLoader(traindata, batch_size=16, shuffle=True)
    testloader = tud.DataLoader(testdata, batch_size=16, shuffle=True)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.003)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_epochs):
        print(f"Epoch {i:>3d}/{num_epochs:>3d}---------------")
        trainloop(trainloader, model, optimizer, loss_fn)
        testloop(testloader, model, loss_fn)
    torch.save(model.state_dict(), "Trained.pth")



if __name__ == "__main__":
    main()

