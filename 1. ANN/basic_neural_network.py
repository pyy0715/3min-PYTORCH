# # 파이토치로 구현하는 신경망
# ## 신경망 모델 구현하기

import torch
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# data 생성
n_dim = 2
x_train, y_train = make_blobs(n_samples=80, n_features=n_dim, centers=[[1,1],[-1,1],[-1,-1],[1,-1]], shuffle=True, cluster_std = 0.3)
x_test, y_test = make_blobs(n_samples=20, n_features=n_dim, centers=[[1,1],[-1,1],[-1,-1],[1,-1]], shuffle=True, cluster_std = 0.3)

y_train = np.where(np.isin(y_train, [0,1]), 0, 1)
y_test = np.where(np.isin(y_test, [0,1]), 0, 1)


def vis_data(x, y, c='r'):
    for x_, y_ in zip(x,y):
        plt.plot(x_[0], x_[1], c+'o' if y_==0 else c+'+')

vis_data(x_train, y_train, c='r')

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        linear1 = self.linear1(input_tensor)
        relu = self.relu(linear1)
        linear2 = self.linear2(relu)
        result = self.sigmoid(linear2)
        return result

model = NeuralNet(2, 5)
learning_rate = 0.05
criterion = nn.BCELoss()

epochs = 2000
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


model.eval()
test_loss_before = criterion(model(x_test).squeeze(), y_test)
print('Before Training, test loss is {}'.format(test_loss_before.item()))

# Trianing
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_output = model(x_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    
    if epoch%100==0:
        print('Trian Loss at {} is {}'.format(epoch, train_loss.item()))

    train_loss.backward()
    optimizer.step()

model.eval()
test_loss_after = criterion(model(x_test).squeeze(), y_test)
print('After Training, test loss is {}'.format(test_loss_after.item()))

# Model Save
torch.save(model.state_dict(),'./model.pt')
print('state_dict format of the model: {}'.format(model.state_dict()))

# Load Model
new_model =  NeuralNet(2,5)
new_model.load_state_dict(torch.load('./model.pt'))

new_model.eval()
print('[-1,1] is label 1을 가질 확률 {}'.format(new_model(torch.FloatTensor([-1,1])).item()))