import torch
import torch.nn as nn
import torch.optim as optim
import string
import unicodedata
import random
import urllib.request
from pathlib import Path
import os


def download_text():
    url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"  # Alice in Wonderland
    file_path = "alice.txt"

    if not Path(file_path).exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, encoding='utf-8') as f:
        text = f.read().lower()

    return ''.join(c for c in text if c in string.ascii_letters + " .,;'")


text_data = download_text()

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def letter_to_index(letter):
    return all_letters.find(letter)


def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_letters)

criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)


def random_training_pair():
    start_idx = random.randint(0, len(text_data) - 6)  # Ensure space for target
    input_line = text_data[start_idx:start_idx + 5]
    target_line = text_data[start_idx + 1:start_idx + 6]
    input_tensor = line_to_tensor(input_line)
    target_tensor = torch.tensor([letter_to_index(c) for c in target_line], dtype=torch.long).unsqueeze(1)
    return input_tensor, target_tensor


n_iters = 500000
every = 50000

for iter in range(1, n_iters + 1):
    input_tensor, target_tensor = random_training_pair()
    hidden = rnn.init_hidden()

    rnn.zero_grad()
    loss = 0

    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])

    loss.backward()
    optimizer.step()

    if iter % every == 0:
        print(f"Iteration {iter}: Loss {loss.item() / input_tensor.size(0)}")


def predict_next_char(input_line):
    with torch.no_grad():
        hidden = rnn.init_hidden()
        input_tensor = line_to_tensor(input_line)

        for i in range(input_tensor.size()[0]):
            output, hidden = rnn(input_tensor[i], hidden)

        topv, topi = output.topk(1)
        return all_letters[topi[0].item()]


print(predict_next_char("Th"))
print(predict_next_char("Al"))