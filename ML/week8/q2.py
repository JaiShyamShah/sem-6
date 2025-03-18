import torch
import torch.nn as nn
import torch.optim as optim
import string
import unicodedata
import random
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os
import urllib.request
import zipfile


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if c in string.ascii_letters + "'-"
    )


def load_data():
    url = "https://download.pytorch.org/tutorial/data.zip"
    zip_path = "data.zip"
    extract_path = ""

    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    extracted_files = list(Path("data/names").glob("*.txt"))
    if not extracted_files:
        raise RuntimeError("Dataset extraction failed. No text files found in 'data/names/'.")

    category_lines = {}
    all_categories = []
    for filename in extracted_files:
        category = filename.stem
        all_categories.append(category)
        with open(filename, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        category_lines[category] = [unicode_to_ascii(line) for line in lines]

    return category_lines, all_categories


category_lines, all_categories = load_data()
n_categories = len(all_categories)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def letter_to_index(letter):
    return all_letters.find(letter)


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
rnn = RNN(n_letters, n_hidden, n_categories)


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_training_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)

n_iters = 100000
every = 5000

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    if iter % every == 0:
        print(f"Iteration {iter}: Loss {loss.item()}")


def predict(input_line):
    with torch.no_grad():
        hidden = rnn.init_hidden()
        line_tensor = line_to_tensor(input_line)

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return category_from_output(output)


print(predict("Alvarez"))
print(predict("Polinka"))
