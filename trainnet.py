import torch
import torch.nn as nn
import torchvision
import torch.utils.data as tud
import pandas as pd
from PIL import Image
import cv2
import mediapipe
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 100

class SpeedDataset(tud.Dataset):

    def __init__(self, path, pddata, transform=None, train=True, n=2):
        self.path = path + "/"
        self.transform = transform
        self.data = pddata
        if train:
            self.data = self.data.iloc[[0,1,2,3,4]]
        else:
            self.data = self.data.iloc[[5]]
        self.data = self.data.reset_index()

        self.totaldata = []
        self.totalcategorical = []
        for i in range(len(self.data)):
            vdata = np.load(self.data.at[i, "pdatapath"])
            vdata = vdata[:,[15,16,23,24,25,26,31,32],:] #900 Maxes the amount of frames per video, [15,16...] selects the landmarks wanted to use


            cdata = np.load(self.data.at[i, "cdatapath"])
            for j in range(n, len(vdata)-n):
                slice = vdata[j-n:j+n]

                if 1 in cdata[j-n:j+n]:
                    self.totalcategorical.append(1)
                else:
                    self.totalcategorical.append(0)

                
                self.totaldata.append(slice)

        self.totalcategorical = np.array(self.totalcategorical)
        self.totaldata = np.array(self.totaldata)
        self.totaldata = self.totaldata.reshape([self.totaldata.shape[0],-1])
                

    def __len__(self):
        return len(self.totalcategorical)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #print(self.totaldata[idx])
        #sample = torch.Tensor(self.totaldata[idx]), torch.tensor(self.totalcategorical[idx], dtype=torch.long)
        category = torch.tensor(self.totalcategorical[idx])
        data = torch.tensor(self.totaldata[idx])
        sample = (data, category)

        if self.transform:
            sample[0] = self.transform(sample[0])
        
        return sample
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(96, 40), #Did 12 12 12 12, trying 17 17 12 12 
            nn.ReLU(),
            nn.Linear(40,2)
        )
        
    
    def forward(self, x):
        #x = x.view(-1)
        #print(x.size())
        #print(x)
        x = self.network(x.float())
        return x


def trainloop(dataloader, model, optimizer, loss_fn, verbose=True):
    size = len(dataloader.dataset)
    #print(f"{len(dataloader)}, Dataldoaer size")
    for batch, (X,y) in enumerate(dataloader):
        y = torch.tensor([1]).long() if 1 in y else torch.tensor([0]).long()
        X, y = X.to(device), y.to(device)
    
        #print(X.size())

        pred = model(X)
        #print(pred.size())

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
    with torch.no_grad():
        for X, y in dataloader:
            y = torch.tensor([1]).long() if 1 in y else torch.tensor([0]).long()
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
    pddata = pd.read_csv("datalabels.csv")
    pddata = pddata.head(6)
    testdata = SpeedDataset(path, pddata, train=False)
    traindata = SpeedDataset(path, pddata, train=True)
    

    trainloader = tud.DataLoader(traindata, batch_size=1, shuffle=False)
    testloader = tud.DataLoader(testdata, batch_size=1, shuffle=False)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_epochs):
        verbose = i % 10 == 0
        if verbose:
            print(f"Epoch {i:>3d}/{num_epochs:>3d}---------------")
        trainloop(trainloader, model, optimizer, loss_fn, False)
        #testloop(testloader, model, loss_fn, verbose)
    torch.save(model.state_dict(), f"saved_models/SGD100e001lr.pt")



if __name__ == "__main__":
    main()

