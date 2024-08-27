
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
# !pip install fastText
import fasttext
import sys
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json

if len(sys.argv) < 2:
    print("give model name as 1st paramater - Dnn , Cnn, Lstm , give 2nd paramater as model path - ./Dnn_model.pickle")
    exit()
filename = sys.argv[1]
file_path = sys.argv[2]

train_data = pd.read_csv('train_split.csv')
val_data = pd.read_csv('val_split.csv')
test_data = pd.read_csv('test_split.csv')

train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']

# Combine train and validation data for fasttext training
combined_data = pd.concat([train_data, val_data])

# Combine train and validation sentences for fasttext training
combined_data_df = pd.concat([train_data, val_data], axis=0)
max_inp_len = max(combined_data_df['tweet'].apply(lambda x: len(x.split(" "))))

# combined_data_df[["tweet"]].to_csv('combined_data.txt', header=False, index=False,  encoding='utf-8')

# # Specify the path for the fasttext model
# fasttext_model_path = 'fasttext_model.bin'

# # Train the FastText model on the combined dataset
# model = fasttext.train_unsupervised('combined_data.txt', model='skipgram')
# model.save_model(fasttext_model_path)

# Specify the path for the fasttext model
fasttext_model_path = 'fasttext_model.bin'

fasttext_model = fasttext.load_model(fasttext_model_path)

import numpy as np

def create_embedding_matrix(sentences, model, max_inp_len):
    embedding_matrix = np.zeros((len(sentences), max_inp_len, model.get_dimension()))

    for i, sentence in enumerate(sentences):
        tokens = sentence.split()[:max_inp_len]
        for j, token in enumerate(tokens):
            embedding_matrix[i, j, :] = model.get_word_vector(token)

    print(embedding_matrix.shape)
    return embedding_matrix

max_inp_len = max(pd.concat([train_data, val_data], axis=0)['tweet'].apply(lambda x: len(x.split(" "))))
d = fasttext_model.get_dimension()
print("max_inp_len , d")
print(max_inp_len , d)

# Create embedding matrices for train, val, and test datasets
# train_embedding_matrix = create_embedding_matrix(train_data['tweet'], fasttext_model, max_inp_len)
# val_embedding_matrix = create_embedding_matrix(val_data['tweet'], fasttext_model, max_inp_len)
test_embedding_matrix = create_embedding_matrix(test_data['tweet'], fasttext_model, max_inp_len)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        # Convert sparse matrix to dense tensor
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Add an extra dimension to represent channels (1 in this case)
        x_sample = self.x[idx]   # .unsqueeze(0)
        return x_sample, self.y[idx]

# train_dataset = CustomDataset(train_embedding_matrix, train_labels)
# eval_dataset = CustomDataset(val_embedding_matrix, val_labels)
test_dataset = CustomDataset(test_embedding_matrix, test_labels)


test_labels.shape , test_embedding_matrix.shape

if not torch.cuda.is_available():
    print("CUDA is not available. Loading model on CPU.")
    map_location = torch.device('cpu')
else:
    map_location = None

device = "cuda" if torch.cuda.is_available() else "cpu"
# device

"""## Dnn"""
import torch.nn.init as init  # Import the init module from PyTorch

class DeepNeuralNetworkO(nn.Module):
  def __init__(self, input_dim = 13704, output_dim = 2):
    super().__init__()
    self.linear_1 = nn.Linear(13704, 256)
    self.activation_1 = nn.ReLU()

    self.linear_2 = nn.Linear(256, 64)
    self.activation_2 = nn.ReLU()

    self.linear_3 = nn.Linear(64, output_dim)

  def forward(self, x):
    out = self.linear_1(x)
    out = self.activation_1(out)

    out = self.linear_2(out)
    out = self.activation_2(out)

    out = self.linear_3(out)

    return out
  
class DeepNeuralNetwork2(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_units, dropout_prob, activation, optimizer_name,weight_init_method, learning_rate):
        super(DeepNeuralNetwork2, self).__init__()

        layers = []
        in_features = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_prob))
            in_features = n_units
        layers.append(nn.Linear(in_features, output_dim))

        self.layers = nn.Sequential(*layers)
        # Initialize optimizer and learning rate scheduler
        self.optimizer_name = optimizer_name
        self.optimizer = self._get_optimizer(optimizer_name, learning_rate)
        self.apply_weight_init(weight_init_method)
        # self.scheduler = self._get_lr_scheduler(lr_schedule_name)

    def _get_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'Invalid optimizer name: {optimizer_name}')

    def apply_weight_init(self, weight_init_method):
        if weight_init_method == 'uniform':
            init_func = init.uniform_
        elif weight_init_method == 'normal':
            init_func = init.normal_
        elif weight_init_method == 'xavier':
            init_func = init.xavier_uniform_
        else:
            raise ValueError(f'Invalid weight initialization method: {weight_init_method}')
        # Apply weight initialization to each linear layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_func(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        return self.layers(x)

"""## Cnn"""
import torch.nn as nn

class ConvolutionalNeuralNetwork0(nn.Module):
    def __init__(self, d, num_classes=2):
        super(ConvolutionalNeuralNetwork0, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=d, out_channels=128, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256 * 119, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

import torch.nn.init as init  # Import the init module from PyTorch

class ConvolutionalNeuralNetwork2(nn.Module):
    def __init__(self, input_dim, d, output_dim, n_layers, n_units, dropout_prob, activation, optimizer_name,weight_init_method, learning_rate, cnn_kernel, cnn_stride, cnn_padding, cnn_channel):
        super(ConvolutionalNeuralNetwork2, self).__init__()
        layers = []

        # self.conv1 = nn.Conv1d(in_channels=d, out_channels=128, kernel_size=5)
        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)

        self.conv1 = nn.Conv1d(in_channels=d, out_channels=cnn_channel, kernel_size=cnn_kernel, stride=cnn_stride, padding=cnn_padding)
        input_dim = self.calculate_conv_output_size(input_dim, cnn_kernel, cnn_stride, cnn_padding)
        input_dim = self.calculate_pool_output_size(input_dim, 2)

        self.conv2 = nn.Conv1d(in_channels=cnn_channel, out_channels=cnn_channel, kernel_size=cnn_kernel, stride=cnn_stride, padding=cnn_padding)
        input_dim = self.calculate_conv_output_size(input_dim, cnn_kernel, cnn_stride, cnn_padding)
        input_dim = self.calculate_pool_output_size(input_dim, 2)

        self.fc1 = nn.Linear(cnn_channel * input_dim, 128)

        self.activation = activation
        self.pool = nn.MaxPool1d(2)
        self.bn = nn.BatchNorm1d(cnn_channel)

        in_features = 128
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_prob))
            in_features = n_units
        layers.append(nn.Linear(in_features, output_dim))
        self.layers = nn.Sequential(*layers)
        # Initialize optimizer and learning rate scheduler
        self.optimizer_name = optimizer_name
        self.optimizer = self._get_optimizer(optimizer_name, learning_rate)
        # self.apply_weight_init(weight_init_method)

    def _get_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'Invalid optimizer name: {optimizer_name}')

    def apply_weight_init(self, weight_init_method):
        if weight_init_method == 'uniform':
            init_func = init.uniform_
        elif weight_init_method == 'normal':
            init_func = init.normal_
        elif weight_init_method == 'xavier':
            init_func = init.xavier_uniform_
        else:
            raise ValueError(f'Invalid weight initialization method: {weight_init_method}')
        # Apply weight initialization to each linear layer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_func(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

    def calculate_pool_output_size(self, input_size, pool_size):
        return int((input_size  / pool_size))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)

        # Flatten the output before the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return self.layers(x)

"""## lstm"""
class LSTM0(nn.Module):
    def __init__(self, input_size, num_layers= 1, output_size =2):
        super(LSTM0, self).__init__()

        self.lstm = nn.LSTM(input_size, 256, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, output_size)  # multiplied by 2 for bidirectional LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        x = self.fc(lstm_out)
        return x

class LSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim =2, num_layers=1, dropout=0.2, bidirectional=False):
        super(LSTM2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Extract the output of the last time step
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        out = self.fc(last_output)

        return out

"""

## for Dnn extra handle
"""

class CustomDatasetDNN(Dataset):
    def __init__(self, X, Y):
        # Convert sparse matrix to dense tensor
        self.x = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(Y.values.flatten(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# for dnn
if filename == "Dnn":
    with open('tfidf_test_vectors_with_labels.pickle', 'rb') as f:
        test_tfidf, _ = pd.read_pickle(f)
    test_dataset = CustomDatasetDNN(test_tfidf, test_labels)
    print( test_tfidf.shape , test_labels.shape)

test_dataloader = DataLoader(test_dataset, batch_size=32)


with open(filename+'_best_params.json', 'r') as f:
    best_params = json.load(f)


print(f"runing evaluation on {filename}")

"""# Task 5"""

models = ["Dnn", "Cnn", "Lstm"]

if filename not in models:
    print('only "Dnn", "Cnn", "Lstm" model allowed')
    exit(1)
try:
    torch.load(file_path, map_location=torch.device(device))
except:
    print('invalid file')
    exit(1)

if best_params['activation'] == "ReLU":
    activation =nn.ReLU()
elif best_params['activation'] == "LeakyReLU":
    activation =nn.LeakyReLU()
elif best_params['activation'] == "PReLU":
    activation =nn.PReLU()


def get_classification_report(model, test_dataloader, device, model_file = "Cnn"):
    model.eval().to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for test_x, test_y in test_dataloader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            
            if model_file == "Cnn":
                outputs = model(test_x.permute(0,2,1))
            else:
                outputs = model(test_x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(test_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(f"{model_file} Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\n")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_file} Accuracy: {accuracy:.4f}")

    # Calculate and print precision, recall, and f1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{model_file} Precision: {precision:.4f}")
    print(f"{model_file} Recall: {recall:.4f}")
    print(f"{model_file} F1-Score: {f1:.4f}")

    # Print the classification report
    print(f"{model_file} Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n")


if filename == "Cnn":
    cnn_kernel = 2 * best_params['cnn_kernel'] + 1
    best_model = ConvolutionalNeuralNetwork0(d)
    best_model2 = ConvolutionalNeuralNetwork2(input_dim = max_inp_len, d = d, output_dim=2, n_layers=best_params['n_layers'],
                            n_units=best_params['n_units'], dropout_prob=best_params['dropout_prob'], activation=activation,
                            optimizer_name=best_params['optimizer'],weight_init_method=best_params['weight_init_method'],
                            learning_rate = best_params['learning_rate'],cnn_kernel = cnn_kernel,
                            cnn_stride = best_params['cnn_stride'], cnn_padding = best_params['cnn_padding'],
                            cnn_channel = best_params['cnn_channel'])
    try:
        best_model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
    except:
        best_model2.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
        best_model = best_model2

    get_classification_report(best_model, test_dataloader, device, model_file = "Cnn")

elif filename == "Dnn":

    best_model = DeepNeuralNetworkO(input_dim=13704)
    best_model2 = DeepNeuralNetwork2(input_dim=13704, output_dim=2, n_layers=best_params['n_layers'],
                            n_units=best_params['n_units'], dropout_prob=best_params['dropout_prob'], activation=activation,
                            optimizer_name=best_params['optimizer'],weight_init_method=best_params['weight_init_method'], learning_rate = best_params['learning_rate'])
    try:
        best_model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
    except:
        best_model2.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
        best_model = best_model2

    get_classification_report(best_model, test_dataloader, device, model_file = "Dnn")

else:

    best_model = LSTM0(input_size=d)
    best_model2 = LSTM2(d, hidden_dim=best_params['hidden_dim'], output_dim =2, num_layers=best_params['num_layers'], dropout=best_params['dropout'], bidirectional=best_params['bidirectional'])
    try:
        best_model.load_state_dict(torch.load('Lstm_model.pth', map_location=torch.device(device)))   
    except:
        best_model2.load_state_dict(torch.load('Lstm_model.pth', map_location=torch.device(device)))
        best_model = best_model2
    
    get_classification_report(best_model, test_dataloader, device,"Lstm")

   