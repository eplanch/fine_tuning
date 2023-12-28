###############################################################################
# Author: Emi Planchon
# Date: 12-28-2023
# Title: Fine Tuning a BERT Classification Model (IAA Side Project)
############################################################################### 

# importing libraries
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                         TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

# loading in the Rotten Tomatoes dataset
dataset = load_dataset("rotten_tomatoes")

# loading in the pretrained tokenizer from the base BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# a short function that tokenizes the text of our datasets
def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding = "max_length", truncation = True)

# applying our tokenizing function onto our train, test, and validation datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# sampling our train data set to just 1000 observations
train_dataset = tokenized_datasets["train"].shuffle(seed = 123).select(range(1000))

# sampling our validation data set to just 1000 observations
eval_dataset = tokenized_datasets["validation"].shuffle(seed = 123).select(range(1000))

# loading the pretrained BERT classification model
pretrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 2)

# creating the training arguments class containing the default hyperparameters
training_args = TrainingArguments(output_dir = "test_trainer", evaluation_strategy = "epoch")

# loading the accuracy function from the Evaluate library
metric = evaluate.load("accuracy")

# a function that calculate the accuracy of the model's predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions=predictions, references=labels)

# creating the trainer object
trainer = Trainer(
    model = pretrained_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
)

# fine tune the model
trainer.train()

# check for classification accuracy on the test set
trainer.evaluate(tokenized_datasets['test'])