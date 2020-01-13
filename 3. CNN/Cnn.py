import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torchvision import transforms, datasets
from torch.utils import data
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu" 


EPOCHS = 40
BATCH_SIZE = 64

transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

train = datasets.FashionMNIST(
    root = './data/',
    train = True,
    download = True,
    transform = transforms
)
test = datasets.FashionMNIST(
    root = './data/',
    train = False,
    download = True,
    transform = transforms
)

train_loader = data.DataLoader(
    dataset = train,
    batch_size = BATCH_SIZE,
    shuffle = True
)
test_loader = data.DataLoader(
    dataset = test,
    batch_size = BATCH_SIZE,
    shuffle = False
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) 
        self.drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2)) 
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = self.drop(x)
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if idx % 200 == 0:
            print('Train Epoch: {} [ {}/{} ({:.0f}%)]\tloss:{:.6f}'.format(
                epoch, idx*len(img), len(train_loader.dataset), 100.*idx / len(train_loader), loss.item()
            ))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct /len(test_loader.dataset)

    return test_loss, test_accuracy

for epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))