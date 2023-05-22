# Trainer script for LoRA finetuning of GPT models
# Run using
#   python -m trainer_lora_finetuning

import os

from datasets.text_dataset import TextDataset
from tokenizers.char_tokenizer import CharTokenizer
from layers.gpt_with_lora_finetuning import GPTWithLoRAFinetuning
from layers import layer_utils
from evaluate import Evaluator

from collections.abc import Callable
import torch, torch.nn as nn


# Define hyperparameters.

# Dataset hyper parameters
data_filename = 'testdata/tinyshakespeare.txt'
train_fraction = 0.9

# Tokenizer hyperparameters.
tokenizer = CharTokenizer(filename = data_filename)

# Architecture hyperparameters.
embedding_dimension = 64
num_heads = 8
head_dimension = 16
num_decoder_blocks = 10

# LoRA finetuning parameters.
mode = 'finetuning'
pretrained_model_path = 'output/gpt_pretrained_model.pt'

# Training hyperparameters.
max_block_size = 24
batch_size = 32
num_batches_to_train = 500

# Evaluation hyperparameters.
num_batches_to_evaluate = 10
num_tokens_to_generate_during_evaluation = 10
num_batches_between_evaluations = 10

# Output parameters.
output_model_path = 'output/gpt_lora_finetuned_model.pt'
output_params_path = 'output/num_parameters_lora_finetuning.txt'

# Fixing seed for reproducing results.
torch.manual_seed(123)

# Create directory corresponding to output_model_path if it does not exist.
model_dirname = os.path.dirname(output_model_path)
if not os.path.exists(model_dirname):
    os.makedirs(model_dirname)

# Generate train and validation datasets and data loaders.
train_dataset = TextDataset(max_block_size, tokenizer, 'train', train_fraction, filename = data_filename)
val_dataset = TextDataset(max_block_size, tokenizer, 'val', train_fraction, filename = data_filename)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True)

# Define the model architecture and optimizer.
model = GPTWithLoRAFinetuning(num_decoder_blocks, tokenizer.vocabulary_length(), embedding_dimension, num_heads, head_dimension, max_block_size, mode=mode, pretrained_model_path = pretrained_model_path)
num_model_parameters = layer_utils.num_parameters(model, output_params_path)
print(f'Number of parameters in the model is {num_model_parameters["total_trainable_parameters"]}.')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
evaluator = Evaluator()

# Perform model training and evaluation.
for (batch_index, train_batch) in enumerate(train_dataloader):
    if batch_index > num_batches_to_train:
        print('Reached maximum number of matches. Training is now complete.')
        torch.save(model.state_dict(), output_model_path)
        break

    train_features = train_batch['features']
    train_labels = train_batch['labels']
    predictions, loss = model(train_features, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % num_batches_between_evaluations == 0:
        (train_loss, val_loss) = evaluator.evaluate_train_and_validation_loss(train_dataloader, val_dataloader, model, num_batches_to_evaluate)
        generated_text = evaluator.generate_text(model, num_tokens_to_generate_during_evaluation, tokenizer)
        print(f' Batch index: {batch_index}, train loss: {train_loss}, val_loss: {val_loss}, generated text\n {generated_text}')
