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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForMaskedLM
import re
# !pip install accelerate
# !pip install transformers

import transformers
import accelerate
# print("Transformers version:", transformers.__version__)
# print("Accelerate version:", accelerate.__version__)

if len(sys.argv) < 3:
    print("please give traincsvr path as 1st argument and val csv path in 2nd argument and ber model name as 3rd argument")
    print("example - python AutoModel.py train_split.csv val_split.csv bert-base-uncased")
    print("available models are-")
    print("bert-base-uncased   bert-base-cased   covid-twitter-bert   twhin-bert-base   SocBERT-base")
    exit(1)

user_model_name_original = sys.argv[3] # "covid-twitter-bert"
train_file = sys.argv[1] # "train_split.csv"
val_file = sys.argv[2] # "val_split.csv"
# test_file = "test_split.csv"

print(f"AutoModelForSequenceClassification for {user_model_name_original}-")

model_list = ["google-bert/bert-base-uncased", "google-bert/bert-base-cased", "digitalepidemiologylab/covid-twitter-bert","Twitter/twhin-bert-base", "sarkerlab/SocBERT-base"]
for model_name in model_list:
    if user_model_name_original == re.split(r'/', model_name)[1]:
        user_model_name = model_name
        break
print(user_model_name)

tokenizer = AutoTokenizer.from_pretrained(user_model_name)
model = AutoModelForSequenceClassification.from_pretrained(user_model_name, num_labels=2)

"""# task 1 , 2 , 3 (load)"""

try:
    # Load train and test data
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    # test_data = pd.read_csv(test_file)

except:
    print("file path are invalid, AutoModel takes csv as input , as pre vector embedding giving max length error")
    # exit(1)


"""# Task 4"""

train_data.head(5)

# Tokenize inputs for train, validation, and test sets with add_special_tokens=True
train_encodings = tokenizer(train_data["tweet"].tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_data["tweet"].tolist(), truncation=True, padding=True, max_length=512)
# test_encodings = tokenizer(test_data["tweet"].tolist(), truncation=True, padding=True, max_length=512)

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

train_dataset = CustomDataset(train_encodings, train_data["label"].tolist())
val_dataset = CustomDataset(val_encodings, val_data["label"].tolist())
# test_dataset = CustomDataset(test_encodings, test_data["label"].tolist())

"""## Basic Network"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# !pip install transformers[torch]
# !pip install accelerate -U

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./my_awesome_model",             # Directory where model checkpoints and logs will be saved
    evaluation_strategy="epoch",                 # Evaluate the model after each epoch
    per_device_train_batch_size=16,              # Batch size per GPU during training
    per_device_eval_batch_size=16,               # Batch size for evaluation
    logging_dir="./logs",                        # Directory for storing logs
    logging_steps=500,                           # Log every 500 steps
    save_strategy="epoch",                       # Save a checkpoint at the end of each epoch
    save_total_limit=3,                          # Maximum number of checkpoints to save
    load_best_model_at_end=True,                 # Load the best model checkpoint at the end of training
    metric_for_best_model="eval_loss",           # Metric to use for determining the best model
    greater_is_better=False,                     # Smaller eval_loss is better
    # Add more configurations as needed
)

# Define Trainer with better configurations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,            # Function to compute evaluation metrics
)

# Train the model
trainer.train()

torch.save(model.state_dict(), "AutoModelForSequenceClassification_"+user_model_name_original+'.pth')
print(f"model saved as AutoModelForSequenceClassification_{user_model_name_original}.pth")
