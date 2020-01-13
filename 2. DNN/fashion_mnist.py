from torchvision import datasets, transforms, utils 
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np


transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
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

batch_size = 32

train_loader = data.DataLoader(
    dataset = train,
    batch_size = batch_size,
    shuffle = True
)
test_loader = data.DataLoader(
    dataset = test,
    batch_size = batch_size,
    shuffle = False
)

data_iter = iter(train_loader)
images, labels = next(data_iter)

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))

img = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10,7))
plt.imshow(np.transpose(npimg,(1,2,0)))
plt.show()

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

for label in labels:
    print('Label is {}\n'.format(CLASS[label.item()]))

import random
idx = random.randint(0,16)

item_img = images[idx]
item_npimg = item_img.numpy()
plt.figure(figsize=(10,7))
plt.title(CLASS[labels[idx].item()])
plt.imshow(item_npimg.squeeze())
plt.show()
