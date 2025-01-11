from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load and preprocess dataset
dataset = load_dataset("text", data_files={"train": "data/dataset.txt"})
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(output_dir="./models", num_train_epochs=3, per_device_train_batch_size=4)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets["train"])
trainer.train()

# Save the fine-tuned model
model.save_pretrained("models/fine_tuned_model")
tokenizer.save_pretrained("models/fine_tuned_model")
