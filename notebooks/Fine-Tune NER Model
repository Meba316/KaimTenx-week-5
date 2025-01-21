from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

# Load the labeled data
dataset = pd.read_csv('labeled_data.csv')  # Your manually labeled CoNLL dataset
# Here we convert it into a Hugging Face dataset format
# Assuming dataset is in DataFrame, convert to dataset
dataset = Dataset.from_pandas(dataset)

# Load pre-trained model (e.g., XLM-RoBERTa)
model_name = "xlm-roberta-base"
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5)  # Adjust based on entities

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(example):
    # Tokenization
    tokenized_inputs = tokenizer(example['text'], padding=True, truncation=True)
    # Align the labels with the tokenized inputs
    return tokenized_inputs

dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # You should ideally have a separate validation dataset
)

trainer.train()
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
