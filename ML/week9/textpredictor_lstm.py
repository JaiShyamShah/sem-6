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

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

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
    optimizer.zero_grad()

    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}')

def predict_next_char(model, input_seq, n_predict=1):
    model.eval()
    predicted_chars = []
    
    for _ in range(n_predict):
        # Prepare the input sequence
        seq_indices = [char_to_index[char] for char in input_seq[-sequence_length:]]
        input_tensor = torch.zeros(1, sequence_length, n_characters)
        for i in range(sequence_length):
            input_tensor[0][i][seq_indices[i]] = 1
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get the most likely character
        top_char_idx = torch.argmax(output).item()
        predicted_char = index_to_char[top_char_idx]
        predicted_chars.append(predicted_char)
        
        # Update input sequence for next prediction
        input_seq += predicted_char
    
    return ''.join(predicted_chars)

input_sequence = "hello worl"  
predicted_chars = predict_next_char(model, input_sequence, n_predict=5)
print(f'Input sequence: "{input_sequence}"')
print(f'Predicted next characters: "{predicted_chars}"')