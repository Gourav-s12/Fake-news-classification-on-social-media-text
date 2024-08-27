# -*- coding: utf-8 -*-
"""EvalTestCustom.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GOqtAptv2JlIob0Kj5t4UdcUzWF4cHKx
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import string
# ! pip install emoji
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
import re
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
from scipy.sparse import hstack
import os

# ! pip install ekphrasis
import ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from tqdm import tqdm

if len(sys.argv) < 5:
    print("model path 1st argument, test path with name as 2nd argument, 3rd type of DL model as cmd args , 4th the bert model type (for vectorizing)")
    print("example - python evaltestcustom.py Dnn_model_bert-base-uncased.pth ./unknown.csv Dnn bert-base-uncased")
    exit()
model_path = sys.argv[1]
user_file_name = sys.argv[2]
type_of_DL_model = sys.argv[3]
bert_model_name = sys.argv[4]

# file_name = "./train_split.csv"
# train_df = pd.read_csv(file_name)

try:
    file_name =  user_file_name 
    test_df = pd.read_csv(file_name)

except:
    print("not a correct csv file")
    exit(1)

# test_df = test_df.iloc[:100]

test_df['label'] = test_df['label'].map({'real': 1, 'fake': 0})

punct_set = set(string.punctuation + '''…'"`’”“'''  + '️')
stoplist = set(nltk.corpus.stopwords.words('english'))

def emoji_split(e, joiner = '\u200d',
                variation_selector=b'\xef\xb8\x8f'.decode('utf-8'),
                return_special_chars = False):
  parts = []
  for part in e:
    if part == joiner:
      if return_special_chars:
        parts.append(":joiner:")
    elif part == variation_selector:
      if return_special_chars:
        parts.append(":variation:")
    else:
      parts.append(part)
  return parts

def data_preprocessor(text):

    # Remove &amp;
    text = text.replace('&amp;', ' ')

    # Remove newline characters
    text = text.replace('\n', ' ')

    # extra space
    text = re.sub('\s+',' ',text)

    # lowercase
    text = text.lower()

    # tokenize
    tt = nltk.tokenize.TweetTokenizer()
    tokens = tt.tokenize(text)

    # Remove stopwords
    tokens = [token for token in tokens if token not in stoplist ]

    # Remove punctuation
    tokens = [token for token in tokens if token not in punct_set]

    # Process tokens
    updated_tokens = []
    lemmatizer = WordNetLemmatizer()

    for t in tokens:
        # Split emoji into components
        if t in emoji.EMOJI_DATA:
            updated_tokens += emoji_split(t)
        # Keep original hashtags and split them into words
        elif t.startswith('#'):
            updated_tokens += hashtag_segmenter.pre_process_doc(t).split(" ")
            # ['<hashtag>']  + hashtag_segmenter.pre_process_doc(t).split(" ") + ['</hashtag>']
        # Replace user mentions with <user>
        elif t.startswith('@'):
            updated_tokens.append('<user>')
        # Replace URLs with <url>
        elif t.startswith('http'):
            updated_tokens.append('<url>')
        else:
            # print(lemmatizer.lemmatize(t))
            updated_tokens.append(lemmatizer.lemmatize(t))

    # de-emojize
    updated_tokens = [emoji.demojize(token) for token in updated_tokens ]

    # Lemmatization
    # lemmatizer = WordNetLemmatizer()
    # updated_tokens = [lemmatizer.lemmatize(token) for token in updated_tokens]

    # print(updated_tokens)
    text = ' '.join(updated_tokens)

    # Replace variations of "co[non-alphanumeric]vid[non-alphanumeric]19" with "covid19"
    text = re.sub(r'co[^\w]*vid', 'covid', text)
    text = re.sub(r'covid[ー^\w]*19', 'covid19', text)

    # clean text
    # text = re.sub(r'\b[^a-zA-Z\s.]+\b', '', text)

    return text


test_df['preprocessed_tweet'] = test_df['tweet'].apply(data_preprocessor)
test_labels = test_df['label']

if type_of_DL_model == "AutoModel":

    print(f"runEval AutoModelForSequenceClassification for {bert_model_name}-")

    # bert_model_name = "SocBERT-base"
    model_list = ["google-bert/bert-base-uncased", "google-bert/bert-base-cased", "digitalepidemiologylab/covid-twitter-bert","Twitter/twhin-bert-base", "sarkerlab/SocBERT-base"]
    for model_name in model_list:
        if bert_model_name == re.split(r'/', model_name)[1]:
            user_model_name = model_name
            break
    print(model_path)

    tokenizer = AutoTokenizer.from_pretrained(user_model_name)
    model_test = AutoModelForSequenceClassification.from_pretrained(user_model_name, num_labels=2)
    # test_labels = test_data["label"].tolist()
    test_encodings = tokenizer(test_df['preprocessed_tweet'].tolist(), truncation=True, padding=True, max_length=512)

    # with open("bert_vectors_test_SocBERT-base.pkl", 'rb') as f:
    #     test_encodings, test_labels = pickle.load(f)
    # Create PyTorch datasets
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    test_dataset = CustomDataset(test_encodings, test_labels)

    print("trainn")
    # Load the model checkpoint and map its tensors to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_test.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    training_args_test = TrainingArguments(
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        output_dir="./results",
    )

    # Define the Trainer instance
    trainer_test = Trainer(
        model=model_test,
        args=training_args_test,
    )

    # Get predictions on the test dataset
    predictions = trainer_test.predict(test_dataset)

    # Extract predicted labels
    y_pred = predictions.predictions.argmax(axis=1)

    y_true = test_dataset[:]['labels']

    print(f"{type_of_DL_model} Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\n")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{type_of_DL_model} Accuracy: {accuracy:.4f}")

    # Calculate and print precision, recall, and f1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{type_of_DL_model} Precision: {precision:.4f}")
    print(f"{type_of_DL_model} Recall: {recall:.4f}")
    print(f"{type_of_DL_model} F1-Score: {f1:.4f}")

    # Print the classification report
    print(f"{type_of_DL_model} Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n")

    exit(1)

model_path_prefix = os.path.dirname(model_path)

with open(model_path_prefix + type_of_DL_model+'_best_params_'+bert_model_name+'.json', 'r') as f:
    best_params = json.load(f)

def process(text_data, model_name, test_labels):
    # Load pre-trained BERT tokenizer
    print(re.split(r'/', model_name)[1])

    if model_name == "sarkerlab/SocBERT-base" or model_name == "Twitter/twhin-bert-base":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    save_model_name = re.split(r'/', model_name)[1]
    # Tokenize input text
    tokenized_text = text_data.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    # Pad sequences to the same length
    max_len = max(map(len, tokenized_text))
    padded_tokenized_text = [text + [0]*(max_len-len(text)) for text in tokenized_text]

    # Convert tokenized text to PyTorch tensors
    input_ids = torch.tensor(padded_tokenized_text)

    # Initialize BERT model
    model = BertModel.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()

    # List to store the vectors
    bert_vectors = []

    # Process each text sample
    for text in tqdm(text_data):
        # Tokenize input text
        tokenized_text = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)

        # Convert tokenized text to PyTorch tensor
        input_ids = torch.tensor(tokenized_text).unsqueeze(0)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)

        # Extract the output representations (vectors) from BERT
        bert_output = outputs[0]  # Output of the last layer

        # Average pooling of the output representations
        pooled_output = torch.mean(bert_output, dim=1).squeeze().numpy()

        # Append the pooled output to the list
        bert_vectors.append(pooled_output)

    # Convert the list of vectors to a numpy array
    bert_vectors = np.array(bert_vectors)

    # Save bert_vectors into pickle file
    # with open("bert_vectors_unknown_"+save_model_name+".pkl", "wb") as f:
    #     pickle.dump((bert_vectors, test_labels), f)

    return bert_vectors

model_list = ["google-bert/bert-base-uncased", "google-bert/bert-base-cased", "digitalepidemiologylab/covid-twitter-bert","Twitter/twhin-bert-base", "sarkerlab/SocBERT-base"]

for model_name in model_list:
    if bert_model_name == re.split(r'/', model_name)[1]:
        # try: # need to remove
        #     with open("bert_vectors_unknown_"+bert_model_name+".pkl", 'rb') as f:
        #         encode_test_vectors, _ = pickle.load(f)
        # except:
            encode_test_vectors = process(test_df['preprocessed_tweet'], model_name, test_labels)

        # with open(checkpoints_path + "bert_vectors_unknown_"+bert_model_name+".pkl", 'rb') as f:
        #     encode_test_vectors, test_labels = pickle.load(f)
# model_path_prefix = os.path.dirname(model_path)

# with open(model_path_prefix + 'bert_vectors_train_'+bert_model_name+'.pkl', 'rb') as f:
#     encode_train_vectors, train_labels = pickle.load(f)

input_dim_x = best_params["input_dim"] # encode_train_vectors.shape[1]
print(f"input_dim_x -{input_dim_x}")



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
    def __init__(self, input_dim , input_channel = 1, num_classes=2):
        super(ConvolutionalNeuralNetwork0, self).__init__()

        self.conv1 = nn.Conv1d(in_channels= input_channel, out_channels=128, kernel_size=5)
        input_dim = self.calculate_conv_output_size(input_dim, 5, 1, 0)
        input_dim = self.calculate_pool_output_size(input_dim, 2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        input_dim = self.calculate_conv_output_size(input_dim, 3, 1, 0)
        input_dim = self.calculate_pool_output_size(input_dim, 2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        input_dim = self.calculate_conv_output_size(input_dim, 3, 1, 0)
        input_dim = self.calculate_pool_output_size(input_dim, 2)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

    def calculate_pool_output_size(self, input_size, pool_size):
        return int((input_size  / pool_size))

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


class ConvolutionalNeuralNetwork2(nn.Module):
    def __init__(self, input_dim, in_channels, output_dim, n_layers, n_units, dropout_prob, activation, optimizer_name,weight_init_method, learning_rate, cnn_kernel, cnn_stride, cnn_padding, cnn_channel):
        super(ConvolutionalNeuralNetwork2, self).__init__()
        layers = []

        # self.conv1 = nn.Conv1d(in_channels=d, out_channels=128, kernel_size=5)
        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_channel, kernel_size=cnn_kernel, stride=cnn_stride, padding=cnn_padding)
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
    
"""

## for model extra handle dataset loader
"""

class CustomDatasetCNN(Dataset):
    def __init__(self, X, Y):
        # Convert sparse matrix to dense tensor
        self.x = torch.tensor(X.reshape(-1, 1, input_dim_x), dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Add an extra dimension to represent channels (1 in this case)
        x_sample = self.x[idx]   # .unsqueeze(0)
        return x_sample, self.y[idx]

class CustomDatasetDNN(Dataset):
    def __init__(self, X, Y):
        # Convert sparse matrix to dense tensor
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y.values.flatten(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
  

if type_of_DL_model == "Dnn":
    test_dataset = CustomDatasetDNN(encode_test_vectors, test_labels)
elif type_of_DL_model == "Cnn":
    test_dataset = CustomDatasetCNN(encode_test_vectors, test_labels) 
else:
    print("wrong model name")
    exit(1)

test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)


print(f"runing evaluation on {type_of_DL_model} {bert_model_name}")

"""# Task 5"""

models = ["Dnn", "Cnn", "Lstm"]

if type_of_DL_model not in models:
    print('only "Dnn", "Cnn", "Lstm" model allowed')
    exit(1)
try:
    torch.load(model_path, map_location=torch.device(device))
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


if type_of_DL_model == "Cnn":
    cnn_kernel = 2 * best_params['cnn_kernel'] + 1
    best_model = ConvolutionalNeuralNetwork0(input_dim_x)
    best_model2 = ConvolutionalNeuralNetwork2(input_dim = input_dim_x, in_channels=1, output_dim=2, n_layers=best_params['n_layers'],
                            n_units=best_params['n_units'], dropout_prob=best_params['dropout_prob'], activation=activation,
                            optimizer_name=best_params['optimizer'],weight_init_method=best_params['weight_init_method'],
                            learning_rate = best_params['learning_rate'],cnn_kernel = cnn_kernel,
                            cnn_stride = best_params['cnn_stride'], cnn_padding = best_params['cnn_padding'],
                            cnn_channel = best_params['cnn_channel'])
    try:
        best_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except:
        best_model2.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        best_model = best_model2

    get_classification_report(best_model, test_dataloader, device,"Cnn")

elif type_of_DL_model == "Dnn":

    best_model = DeepNeuralNetworkO(input_dim_x)
    best_model2 = DeepNeuralNetwork2(input_dim=input_dim_x, output_dim=2, n_layers=best_params['n_layers'],
                            n_units=best_params['n_units'], dropout_prob=best_params['dropout_prob'], activation=activation,
                            optimizer_name=best_params['optimizer'],weight_init_method=best_params['weight_init_method'], learning_rate = best_params['learning_rate'])
    try:
        best_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except:
        best_model2.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        best_model = best_model2

    get_classification_report(best_model, test_dataloader, device,"Dnn")

else:
    pass
   