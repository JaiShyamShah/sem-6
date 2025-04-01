import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

text = "hello world! this is a simple character prediction model using LSTM. let's see how it performs."

all_characters = sorted(list(set(text)))
n_characters = len(all_characters)
char_to_index = {char: idx for idx, char in enumerate(all_characters)}
index_to_char = {idx: char for idx, char in enumerate(all_characters)}

n_hidden = 128
n_layers = 2
learning_rate = 0.005
sequence_length = 10
n_iters = 1000


class LSTMCharacterPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCharacterPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(n_layers, batch_size, self.hidden_size),
                torch.zeros(n_layers, batch_size, self.hidden_size))


def prepare_sequences(text, seq_length):
    inputs = []
    targets = []
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        target = text[i + seq_length]
        inputs.append([char_to_index[char] for char in seq])
        targets.append(char_to_index[target])
    return np.array(inputs), np.array(targets)


X, y = prepare_sequences(text, sequence_length)

X_tensor = torch.zeros(X.shape[0], sequence_length, n_characters)
y_tensor = torch.tensor(y, dtype=torch.long)

for i in range(X.shape[0]):
    for j in range(sequence_length):
        X_tensor[i][j][X[i][j]] = 1

model = LSTMCharacterPredictor(n_characters, n_hidden, n_characters)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(n_iters):
    hidden = model.init_hidden(X_tensor.size(0))
    optimizer.zero_grad()

    output, hidden = model(X_tensor, hidden)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}')


def predict_next_char(model, input_seq, n_predict=1):
    model.eval()
    input_seq = [char_to_index[char] for char in input_seq]
    input_tensor = torch.zeros(1, sequence_length, n_characters)

    for i in range(sequence_length):
        input_tensor[0][i][input_seq[i]] = 1

    hidden = model.init_hidden(1)
    predicted_chars = []

    for _ in range(n_predict):
        output, hidden = model(input_tensor, hidden)
        top_char_idx = torch.argmax(output).item()
        predicted_chars.append(index_to_char[top_char_idx])

        input_seq = input_seq[1:] + [top_char_idx]
        input_tensor = torch.zeros(1, sequence_length, n_characters)
        for i in range(sequence_length):
            input_tensor[0][i][input_seq[i]] = 1

    return ''.join(predicted_chars)


input_sequence = "hello worl"
predicted_char = predict_next_char(model, input_sequence, n_predict=5)
print(f'Input sequence: "{input_sequence}"')
print(f'Predicted next characters: "{predicted_char}"')