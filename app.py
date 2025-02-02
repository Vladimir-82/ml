"""Запуск бота."""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

tokenizer = AutoTokenizer.from_pretrained('./my_model')
model = AutoModelForCausalLM.from_pretrained('./my_model')


def generate_response(request):
    """Генерация ответа."""
    input = tokenizer.encode(request, return_tensors='pt')
    attention_mask = torch.ones(input.shape, dtype=torch.long)
    outputs = model.generate(input, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input)
        print('Bot:', response)
