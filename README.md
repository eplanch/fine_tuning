# Fine Tuning a BERT Classification Model

In this repository, I document and walkthrough my process of fine tuning a pretrained BERT classification model to complete sentiment analysis on a Rotten Tomatoes dataset. I wanted to use this project as an opporunity to better familiarize myself with Hugging Face and their Transformers library. Similarly, I was eager to gain some experience in fine tuning some popular transformer models, as this framework can be applied to different types of langauge models, including GPT2 and GPT3. 

As I mentioned above, I'll provide a notebook-style walkthrough of my work below. The complete python script that I used can also be found in the repository. 

***

The first step in this project was to find a dataset that would allow us to explore the intricaces of fine-tuning models without much complication. We chose the Rotten Tomatoes dataset that Hugging Face provides:

```
# loading in the Rotten Tomatoes dataset
dataset = load_dataset("rotten_tomatoes")
```

Now that the dataset was found, we preprocess the data into the expected model input format. The Transformers library provides pretrained tokenizers that allows us to convert our text data into a sequence of tokens, and then into a sequence of embeddings.

Note that we pad and truncate our individual reviews to create even-lengthed sequences for the model to be able to train on them. We also used the `map` function to tokenize both our training and our test dataset.

In an effort to reduce training time, we also randomly sampled 1000 of the 8530 reviews in our training and validation sets. 

```
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
```

Once the data was formatted correctly, we loaded in a pretrained BERT classification model (that is trained on predicting two classes). 

```
# loading the pretrained BERT classification model
pretrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 2)
```

After our pretrained model has been loaded and our data has been formatted correctly, we began to focus on fine tuning the pretrained classification model. We used the (PyTorch) `Trainer` class within the Transformers library to fine tune our model. Beofre tuning the model, we created a `TrainingArguments` class using the default training hyperparameters. 

```
# creating the training arguments class containing the default hyperparameters
training_args = TrainingArguments(output_dir = "test_trainer", evaluation_strategy = "epoch")
```

The `Trainer` class does not automatically evaluate model performance during training, so we created a function that computes evaluation metrics and pass it into the `Trainer` class:

```
# loading the accuracy function from the Evaluate library
metric = evaluate.load("accuracy")

# a function that calculate the accuracy of the model's predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions=predictions, references=labels)
```

At this point, we were able to create our `Trainer` object with the pretrained model, the training arguements, the training and test datasets, and the `compute_metrics` function we created above:

```
# creating the trainer object
trainer = Trainer(
    model = pretrained_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
)
```

Finally, we fine tuned the model:

```
trainer.train()
```

***

Once the fine tuned model had been produced, we calculated the classification accuracy of the fine tuned model on the full test set:

```
# check for classification accuracy of the fine tuned model on the test set
trainer.evaluate(tokenized_datasets['test'])
```

Interestingly, we found that the fine tuned model had a classification accuracy of 81.8%! Given the sometimes confusing format of the Rotten Tomatoes reviews, we found a lot of benefit in fine tuning a pretrained BERT classifier, namely the low amount of preprocessing needed to achieve such a high accuracy rate. In the future, I'd be interested in looking into fine tuning different types of models (that are not classifiers), including text-generation models such as GPT2 and GPT3. 
