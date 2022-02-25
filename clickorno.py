from errno import ENETRESET
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as tud
import pandas as pd
from PIL import Image

class SpeedDataset(tud.Dataset):

    def __init__(self, path, csv_file, transform=None, train=True, split=.20):
        self.path = path + "/"
        self.transform = transform

        self.data = pd.read_csv(self.path + csv_file)
        if train:
            self.data = self.data.iloc[int(len(self.data) * split):]
        else:
            self.data = self.data.iloc[:int(len(self.data) * split)]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.path + self.data.iloc[idx, 0]

        input = torchvision.transforms.ToTensor()(Image.open(fname))
        category = self.data.iloc[idx, 1]
        sample = (torch.Tensor(input), torch.tensor(category, dtype=torch.long))

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Conv2d(3, 6, 10),
            nn.ReLU(),
            nn.MaxPool2d(10,10),
            nn.Conv2d(6, 16, 10),
            nn.ReLU(),
            nn.MaxPool2d(10,10),
            nn.Flatten(),
            nn.Linear(2592, 1024),
            nn.ReLU(),
            nn.Linear(1024,500),
            nn.ReLU(),
            nn.Linear(500,2),
            nn.Softmax()
        )
        
    
    def forward(self, x):
        x = self.network(x)
        return x


def trainloop(dataloader, model, optimizer, loss_fn):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"Loss: {loss.item():>7f}, [{batch * len(X):>5d}/{size:>5d}]")

def main():
    path = r"C:/Users/erik/OneDrive/Cross-Code/VSCode/PythonCode/Starting_Code/Machine Learning/pytorchlearning/Speed"
    traindata = SpeedDataset(path, "datalabels.csv", train=True)
    testdata = SpeedDataset(path, "datalabels.csv", train=False)

    trainloader = tud.DataLoader(traindata, batch_size=16, shuffle=True)
    testloader = tud.DataLoader(testdata, batch_size=16, shuffle=True)

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        trainloop(trainloader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()

