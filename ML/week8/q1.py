import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

df = pd.read_csv("https://datahub.io/core/natural-gas/r/daily.csv")
df = df.dropna()

y = df['Price'].values
x = np.arange(1, len(y) + 1, 1)

minm, maxm = y.min(), y.max()
y = (y - minm) / (maxm - minm)

Sequence_Length = 10
X, Y = [], []
for i in range(len(y) - Sequence_Length):
    X.append(y[i:i + Sequence_Length])
    Y.append(y[i + Sequence_Length])

X, Y = np.array(X), np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=False)


class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output


model = RNNModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 1500
for i in range(epochs):
    for data in train_loader:
        x_batch, y_batch = data
        y_pred = model(x_batch.view(-1, Sequence_Length, 1)).reshape(-1)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 50 == 0:
        print(f"Epoch {i}: Loss = {loss.item()}")

test_set = NGTimeSeries(x_test, y_test)
test_pred = model(test_set[:][0].view(-1, Sequence_Length, 1)).view(-1).detach().numpy()

plt.plot(test_pred, label='Predicted')
plt.plot(test_set[:][1].view(-1), label='Original')
plt.legend()
plt.show()
