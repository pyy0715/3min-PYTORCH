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
    
view_data = trainset.data[:5].view(-1,28*28)
view_data = view_data.type(torch.FloatTensor)/255.

def train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, 28*28).to(device)
        y = x.view(-1, 28*28).to(device)
        label = label.to(device)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(1, EPOCHS+1):
    train(autoencoder, train_loader)
    test_x = view_data.to(device)
    _, decoded_data = autoencoder(test_x)

    f, a = plt.subplots(2, 5, figsize=(5,2))
    print("[Epoch {}]".format(epoch))

    for i in range(5):
        img = np.reshape(view_data.data.numpy()[i], (28,28))
        a[0][i].imshow(img, cmap='gray')

    for i in range(5):
        img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28,28))
        a[1][i].imshow(img, cmap='gray')

    plt.show()

# 잠재 변수를 3D 플롯으로 시각화
view_data = trainset.data[:200].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor)/255
test_x = view_data.to(device)
encoded_data, _ = autoencoder(test_x)
encoded_data = encoded_data.to('cpu')

CLASS = {
    0: 'T-Shirt/Top',
    1: 'Trouser',
    2: 'Pullober',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)

X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()

labels = trainset.targets[:200].numpy()

for x,y,z,s in zip(X,Y,Z,labels):
    name = CLASS[s]
    color = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, name, backgroundcolor=color)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()