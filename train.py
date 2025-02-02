"""Тренировка модели."""

import torch
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
)

model_name = './Mistral-7B-Instruct-v0.1'
tokenizer = transformers.MistralModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.MistralModel.from_pretrained(model_name)


def get_data():
    """Получение данных для обучения."""
    with open('input_dir/input.txt', 'r', encoding="utf-8") as file:
        return file.read().strip()


class CustomDataset(torch.utils.data.Dataset):
    """Данные для тренировки."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        """Получение значения."""
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


dataset = get_data()
train_encodings = tokenizer(dataset, truncation=True, padding=True, return_tensors="pt")
dataset = CustomDataset(train_encodings)

training_data = TrainingArguments(
    output_dir='./my_model',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
)


def train_my_model():
    """Тренировать модель."""
    trainer = Trainer(
        model=model,
        args=training_data,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained('./my_model')
    tokenizer.save_pretrained('./my_model')
