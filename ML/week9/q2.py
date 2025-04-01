import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import string
import unicodedata
import random

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters + " .,;'"
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []
for filename in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input.view(1, -1, n_letters), hidden)
        output = self.fc(self.dropout(output[:, -1, :]))
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size),
                torch.zeros(2, 1, self.hidden_size))

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomTrainingExample():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    return category, line

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def targetTensor(category_index):
    return torch.tensor([category_index], dtype=torch.long)

def letterToIndex(letter):
    return all_letters.find(letter)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

n_hidden = 256
lstm = LSTMClassifier(n_letters, n_hidden, n_categories)
learning_rate = 0.001
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

n_iters = 200000
print_every = 500
plot_every = 100

current_loss = 0
all_losses = []

best_loss = float('inf')
patience = 20000
patience_counter = 0

def train(category_tensor, line_tensor):
    hidden = lstm.initHidden()
    optimizer.zero_grad()

    output, hidden = lstm(line_tensor, hidden)
    output = output.view(1, -1)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

for iter in range(1, n_iters + 1):
    category, line = randomTrainingExample()
    category_tensor = targetTensor(all_categories.index(category))
    line_tensor = inputTensor(line)
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% %.4f %s -> %s %s' % (iter, iter / n_iters * 100, loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping at iteration:", iter)
            break

def evaluate(line_tensor):
    hidden = lstm.initHidden()
    output, hidden = lstm(line_tensor, hidden)
    return output.view(1, -1)

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(inputTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')