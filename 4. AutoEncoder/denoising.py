import torch
import torchvision
import torch.nn.functional as F 
from torch import nn, optim
from torch.utils import data
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCHS = 10
BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu" 

trainset = datasets.FashionMNIST(
    root = './data/',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)
testset = datasets.FashionMNIST(
    root = './data/',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

train_loader = data.DataLoader(
    dataset = trainset,
    batch_size = BATCH_SIZE,
    shuffle = True
)
test_loader = data.DataLoader(
    dataset = testset,
    batch_size = BATCH_SIZE,
    shuffle = False
)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3) #마지막 출력은 3차원에서 시각화 할 수 있도록 3개만 남김.
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()
    
def add_noise(img):
    noise = torch.rand(img.size())*0.2
    noisy_img = img + noise
    return noisy_img

def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for step, (x, label) in enumerate(train_loader):
        noisy_x = add_noise(x)  
        noisy_x = noisy_x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(device)
        label = label.to(device)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    return avg_loss / len(train_loader)

for epoch in range(1, EPOCHS+1):
    loss = train(autoencoder, train_loader)
    print("[EPOCH {}] loss:{}".format(epoch, loss))

sample_data = testset.data[0].view(-1, 28*28)
sample_data = sample_data.type(torch.FloatTensor)/255.

original_x = sample_data[0]
noisy_x = add_noise(original_x).to(device)
_, recovered_x = autoencoder(noisy_x)

f, a = plt.subplots(1, 3, figsize=(15,15))
original_img = np.reshape(original_x.to("cpu").data.numpy(), (28,28))
noisy_img = np.reshape(noisy_x.to("cpu").data.numpy(), (28,28))
recovered_img = np.reshape(recovered_x.to("cpu").data.numpy(), (28,28))

a[0].set_title('Original')
a[0].imshow(original_img, cmap='gray')

a[1].set_title('Noisy')
a[1].imshow(noisy_img, cmap='gray')

a[2].set_title('Recovered')
a[2].imshow(recovered_img, cmap='gray')