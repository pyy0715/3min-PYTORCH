import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torchvision import transforms, datasets
from torch.utils import data
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu" 


transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5 ,0.5, 0.5))
    ])

EPOCHS = 300
BATCH_SIZE = 128

train = datasets.CIFAR10(
    root = './data/',
    train = True,
    download = True,
    transform = transforms
)
test = datasets.CIFAR10(
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

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

model = ResNet().to(device)
optimizer = optim.SGD(model.parameters(), lr= 0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
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
    scheduler.step()
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))