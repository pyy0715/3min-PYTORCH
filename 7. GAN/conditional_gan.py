import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('My Device is {}'.format(device))

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST(
    './data',
    train =  True,
    download = True, 
    transform = trans)

train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = opt.batch_size,
    shuffle = True,
    drop_last = True
)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            *self._make_layer(opt.batch_size+10, 256),
            *self._make_layer(256, 512),
            *self._make_layer(512, 1024),
            *self._make_layer(1024, 784, True)
        )

    def forward(self, z, labels):
        c = self.embed(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out

    def _make_layer(self, in_size, out_size, last=False):
        layers = [nn.Linear(in_size, out_size)]
        if last:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)  


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.embed = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            *self._make_layer(794, 1024),
            *self._make_layer(1024, 512),
            *self._make_layer(512, 256),
            *self._make_layer(256, 1, True)
        )

    def forward(self, x, labels):
        c = self.embed(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out

    def _make_layer(self, in_size, out_size, last=False):
        layers = [nn.Linear(in_size, out_size)]
        if last:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
        return nn.Sequential(*layers)  

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

G_optimizer = optim.Adam(G.parameters(), lr=opt.lr)
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr)

best_val_loss = 0
for epoch in range(opt.n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(opt.batch_size, -1).to(device)
        labels = labels.to(device)

        real_labels = torch.ones(opt.batch_size, 1).to(device)
        fake_labels = torch.zeros(opt.batch_size, 1).to(device)

        # Discriminator
        outputs = D(images, labels)
        D_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(opt.batch_size, 128).to(device)
        z_label = torch.randint(0, 10, (opt.batch_size, )).to(device)
        fake_images = G(z, z_label)

        outputs = D(fake_images, z_label)
        D_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        D_loss_total = D_loss_real + D_loss_fake

        # Backpropagation
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()
        D_loss_total.backward(retain_graph=True)
        D_optimizer.step()

        # Generator
        fake_images = G(z, z_label)
        ouputs = D(fake_images, z_label)
        G_loss = criterion(outputs, real_labels)

        # Backpropagation
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    print('Epoch:[{}/{}], D_loss_total: {:.4f}, G_loss: {:.4f}, D(x):{:.2f}, D(G(z)): {:.2f}'.format(
        epoch+1, opt.n_epochs, D_loss_total.item(), G_loss.item(), real_score.mean().item(), fake_score.mean().item()
    ))

    # Check Point
    if best_val_loss == 0 or fake_score.mean().item() >= best_val_loss:
        if not os.path.isdir("checkpoint"):
            os.makedirs("checkpoint")

        torch.save(G.state_dict(), './checkpoint/cycle_g.pt')
        torch.save(D.state_dict(), './checkpoint/cycle_d.pt')
        
        print('###### Model Save At Epoch: {}'.format(epoch+1))
        best_val_loss = fake_score.mean().item()

# Visualization
G.load_state_dict(torch.load('./checkpoint/cycle_g.pt'))

item_number = 9 #boot
z = torch.randn(1, 128).to(device)
z_label = torch.full((1,), 9, dtype = torch.long).to(device)

sample_images = G(z, z_label)
sample_images_img = np.reshape(sample_images.data.cpu().numpy()[0], (28,28))
plt.imshow(sample_images_img, cmap='gray')
plt.show()