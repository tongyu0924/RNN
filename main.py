import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

num_epochs = 1
batch_size = 64
time_step = 28
input_step = 28
lr = 0.01

train_data = pd.read_csv("C:/Users/user/Desktop/data/Digit Recognizer  - train.csv")
print(train_data.isnull().sum().sum())
x_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28).astype("float32")
y_train = train_data.label.values

features_train, features_test, targets_train, targets_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

plt.imshow(x_train[10])
plt.axis("off")
plt.title(str(y_train[10]))
plt.savefig("graph.png")
plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = 28,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, c_n) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = rnn(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("Iteration:{}. Loss:{}. Accuracy:{}".format(step, loss.item(), accuracy))



