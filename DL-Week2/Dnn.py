

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
# ! pip install optuna
import optuna
import sys

if len(sys.argv) < 3:
    print("please give train vector path as 1st argument and test vector path in 2nd argument and ber model name as 3rd argument")
    print("example - python Dnn.py bert_vectors_train_bert-base-uncased.pkl bert_vectors_val_bert-base-uncased.pkl bert-base-uncased")
    print("available models are-")
    print("bert-base-uncased   bert-base-cased   covid-twitter-bert   twhin-bert-base   SocBERT-base")
    exit(1)

model_name = sys.argv[3]
train_file = sys.argv[1]
val_file = sys.argv[2]
print(f"Dnn for {model_name}-")

"""# task 1 , 2 , 3 (load)"""

try:
    with open(train_file, 'rb') as f:
        encode_train_vectors, train_labels = pickle.load(f)

    with open(val_file, 'rb') as f:
        encode_val_vectors, val_labels = pickle.load(f)
except:
    print("file path are invalid")
    exit(1)

print(train_labels.values.shape , encode_train_vectors.shape)
# train_labels = train_labels.values.flatten()
# val_labels  = val_labels.values.flatten()
input_dim_x = encode_train_vectors.shape[1]
print("input dimention")
print(input_dim_x)

"""# Task 4"""

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        # Convert sparse matrix to dense tensor
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y.values.flatten(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = CustomDataset(encode_train_vectors, train_labels)
eval_dataset = CustomDataset(encode_val_vectors, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
eval_dataloader = DataLoader(eval_dataset, batch_size = 32, shuffle = False)

for x, y in train_dataloader:
    print(x.shape, y.shape)
    break

"""## Basic Network"""

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# if not torch.cuda.is_available():
#     raise RuntimeError("CUDA is not available.")

class DeepNeuralNetworkO(nn.Module):
  def __init__(self, input_dim = 768, output_dim = 2):
    super().__init__()
    self.linear_1 = nn.Linear(input_dim, 256)
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

import torch
network = DeepNeuralNetworkO(input_dim_x).to(device)

# from torchsummary import summary
# summary(network, (1, input_dim_x))

# batch_x, batch_y = next(iter(train_dataloader))
# print(batch_x.shape, batch_y.shape)

"""## Basic Training Loop"""

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(network.parameters(), lr = 0.001)
epochs = 10

train_epoch_loss = []
eval_epoch_loss = []

for epoch in tqdm(range(epochs)):
    curr_loss = 0
    total = 0
    for train_x, train_y in train_dataloader:

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        optim.zero_grad()

        y_pred = network(train_x)
        loss = criterion(y_pred, train_y)

        loss.backward()
        optim.step()

        curr_loss += loss.item()
        total += len(train_y)
    train_epoch_loss.append(curr_loss / total)

    curr_loss = 0
    total = 0
    for eval_x, eval_y in eval_dataloader:
        eval_x = eval_x.to(device)
        eval_y = eval_y.to(device)
        optim.zero_grad()

        with torch.no_grad():
            y_pred = network(eval_x)

        loss = criterion(y_pred, eval_y)

        curr_loss += loss.item()
        total += len(train_y)
    eval_epoch_loss.append(curr_loss / total)

# import matplotlib.pyplot as plt

# plt.plot(range(epochs), train_epoch_loss, label='train')
# plt.plot(range(epochs), eval_epoch_loss, label='eval')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

correct = 0
total = 0
for x, y in eval_dataloader:
    x = x.to(device)
    with torch.no_grad():
        yp = network(x)
    yp = torch.argmax(yp.cpu(), dim = 1)
    correct += (yp == y).sum()
    total += len(y)
prev_eval_acc = correct / total
print(f"Accuracy on Eval Data {(prev_eval_acc * 100):.2f}")

prev_model = network

"""## now hyperparameter optimization"""

# !pip install torchmetrics
# from torchmetrics import Accuracy

import torch.nn.init as init  # Import the init module from PyTorch

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

def train_model(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Calculate accuracy
        y_pred = torch.argmax(outputs, dim = 1)
        correct_predictions += (y_pred == targets).sum().item()
        total_samples += len(targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    accuracy = correct_predictions / total_samples
    total_loss /= total_samples
    return total_loss, accuracy

def evaluate_model(network, criterion, eval_dataloader, device):
    network.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for eval_x, eval_y in eval_dataloader:
            eval_x = eval_x.to(device)
            eval_y = eval_y.to(device)

            y_pred = network(eval_x)
            loss = criterion(y_pred, eval_y)

            total_loss += loss.item()

            # Calculate accuracy
            y_pred = torch.argmax(y_pred, dim = 1)
            correct_predictions += (y_pred == eval_y).sum().item()
            total_samples += len(eval_y)

    total_loss /= total_samples
    accuracy = correct_predictions / total_samples

    return total_loss, accuracy

import optuna
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.nn.init import xavier_uniform_, kaiming_normal_
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.nn import LeakyReLU, PReLU
from torch.utils.data import DataLoader
from torch.nn import L1Loss, MSELoss

def objective(trial):
    # Define hyperparameters to be optimized
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 2, 10)
    n_units = trial.suggest_int("n_units", 16, 512)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "PReLU"])
    weight_init_method = trial.suggest_categorical("weight_init_method", ["uniform", "normal", "xavier"])
    optimizer_choice = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    # lr_schedule = trial.suggest_categorical("lr_schedule", ["step_lr", "exp_lr"])
    use_early_stopping = trial.suggest_categorical("use_early_stopping", [True, False])
    patience = trial.suggest_int("patience", 5, 20)
    # gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 5)
    # regularization_strength = trial.suggest_float("regularization_strength", 0.0, 0.1)
    epochs = trial.suggest_int("epochs", 10, 40)

    # Wrap activation functions inside nn.Module subclass
    if activation == "ReLU":
        activation =nn.ReLU()
    elif activation == "LeakyReLU":
        activation =nn.LeakyReLU()
    elif activation == "PReLU":
        activation =nn.PReLU()
    else:
        raise ValueError(f'Invalid activation method: {activation}')

    # Create an instance of your Network
    network = DeepNeuralNetwork2(input_dim=input_dim_x, output_dim=2, n_layers=n_layers,
                            n_units=n_units, dropout_prob=dropout_prob, activation=activation,
                            optimizer_name=optimizer_choice,weight_init_method=weight_init_method, learning_rate = learning_rate)

    optimizer = network.optimizer

    # Move the model to the appropriate device
    network.to(device)
    best_eval_loss = float('inf')
    no_improvement = 0

    # Define DataLoader instances
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss , train_acc = train_model(network, criterion, optimizer, train_dataloader, device)
        eval_loss, eval_acc = evaluate_model(network, criterion, eval_dataloader, device)

        # Report the validation loss to Optuna for optimization
        trial.report(eval_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if use_early_stopping:
            if eval_loss < best_eval_loss:
                best_val_loss = eval_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break

    print(f"Training acc = {train_acc} , Val acc = {eval_acc}")
    return eval_acc  # eval_loss

# Create Optuna study and run optimization
study = optuna.create_study(direction="maximize")  # or "minimize" for a loss
study.optimize(objective, n_trials=15, timeout=2500)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

train_epoch_loss = []
eval_epoch_loss = []

# Retrieve the best parameters
best_params = study.best_params
best_params["input_dim"] = input_dim_x
# Wrap activation functions inside nn.Module subclass
if best_params['activation'] == "ReLU":
    activation =nn.ReLU()
elif best_params['activation'] == "LeakyReLU":
    activation =nn.LeakyReLU()
elif best_params['activation'] == "PReLU":
    activation =nn.PReLU()

# Create an instance of your Network
network2 = DeepNeuralNetwork2(input_dim=input_dim_x, output_dim=2, n_layers=best_params['n_layers'],
                        n_units=best_params['n_units'], dropout_prob=best_params['dropout_prob'], activation=activation,
                        optimizer_name=best_params['optimizer'],weight_init_method=best_params['weight_init_method'], learning_rate = best_params['learning_rate'])

optimizer = network2.optimizer

epochs = best_params['epochs']
hyp_accuracy=study.best_value
cond=True

# if hyp_accuracy < prev_eval_acc:
#     cond=False
#     network2=DeepNeuralNetworkO(input_dim_x , 2)
#     best_params['use_early_stopping'] = False
#     epochs=15
#     optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
# # Learning rate scheduler
# if lr_schedule == "step_lr":
#     scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
# elif lr_schedule == "exp_lr":
#     scheduler = ExponentialLR(optimizer, gamma=0.9)

# Move the model to the appropriate device
network2.to(device)
best_eval_loss = float('inf')
no_improvement = 0
batch_size = best_params['batch_size']

# Define DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Training loop
now_eval_acc = 0
for epoch in tqdm(range(epochs), desc="Epochs"):
    train_loss , train_acc = train_model(network2, criterion, optimizer, train_dataloader, device)
    eval_loss, eval_acc = evaluate_model(network2, criterion, eval_dataloader, device)

    train_epoch_loss.append(train_loss)
    eval_epoch_loss.append(eval_loss)
    now_eval_acc = eval_acc
    # Early stopping
    if best_params['use_early_stopping']:
        if eval_loss < best_eval_loss:
            best_val_loss = eval_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= best_params['patience']:
                break

# import matplotlib.pyplot as plt

# plt.plot(range(epochs), train_epoch_loss, label='train')
# plt.plot(range(epochs), eval_epoch_loss, label='eval')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

test_loss, test_acc = evaluate_model(network2, criterion, eval_dataloader, device)
print(f"Accuracy on Eval Data {test_acc}")

# save model
filename = "Dnn_model_"+model_name+".pickle"

if prev_eval_acc > now_eval_acc:
  print("p")
  # pickle.dump(prev_model, open(filename, "wb"))
  torch.save(prev_model.state_dict(),  'Dnn_model_'+model_name+'.pth')
else:
  # pickle.dump(network2, open(filename, "wb"))
  torch.save(network2.state_dict(), 'Dnn_model_'+model_name+'.pth')

test_loss, test_acc = evaluate_model(network2, criterion, eval_dataloader, device)
print(f"Accuracy on Eval Data {test_acc}")

import json

torch.save(network2.state_dict(), 'Dnn_model_'+model_name+'.pth')
# Assuming best_params is defined
with open('Dnn_best_params_'+model_name+'.json', 'w') as f:
    json.dump(best_params, f)

# with open('bert_vectors_test_'+model_name+'.pkl', 'rb') as f:
#     encode_test_vectors, test_labels = pickle.load(f)
# test_dataset = CustomDataset(encode_test_vectors, test_labels)
# test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# from sklearn.metrics import accuracy_score, classification_report

# def get_classification_report(model, test_dataloader, device, model_file = "Cnn"):
#     model.eval().to(device)
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for test_x, test_y in test_dataloader:
#             test_x, test_y = test_x.to(device), test_y.to(device)

#             if model_file == "Cnn":
#                 outputs = model(test_x.permute(0,2,1))
#             else:
#                 outputs = model(test_x)
#             _, predicted = torch.max(outputs.data, 1)
#             y_true.extend(test_y.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())

#     accuracy = accuracy_score(y_true, y_pred)
#     print(f"{model_file} Accuracy: {accuracy:.4f}")

#     # Calculate and print precision, recall, and f1-score
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')

#     print(f"{model_file} Precision: {precision:.4f}")
#     print(f"{model_file} Recall: {recall:.4f}")
#     print(f"{model_file} F1-Score: {f1:.4f}")

#     # Print the classification report
#     print(f"{model_file} Classification Report:")
#     print(classification_report(y_true, y_pred))
#     print("\n")

# with open('Dnn_best_params_'+model_name+'.json', 'r') as f:
#     best_params = json.load(f)

# best_model = DeepNeuralNetworkO(input_dim_x)
# best_model2 = DeepNeuralNetwork2(input_dim=input_dim_x, output_dim=2, n_layers=best_params['n_layers'],
#                         n_units=best_params['n_units'], dropout_prob=best_params['dropout_prob'], activation=activation,
#                         optimizer_name=best_params['optimizer'],weight_init_method=best_params['weight_init_method'], learning_rate = best_params['learning_rate'])
# try:
#     best_model.load_state_dict(torch.load('Dnn_model_'+model_name+'.pth', map_location=torch.device(device)))
#     get_classification_report(best_model, test_dataloader, device,"Dnn")
# except:
#     best_model2.load_state_dict(torch.load('Dnn_model_'+model_name+'.pth', map_location=torch.device(device)))
#     report = get_classification_report(best_model2, test_dataloader, device,"Dnn")
#     best_model = best_model2