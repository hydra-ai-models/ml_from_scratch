# Python script to train CharGPT models.

from datasets.text_dataset import TextDataset
from tokenizers.char_tokenizer import CharTokenizer
from layers.gpt import GPT
from evaluate import Evaluator

from collections.abc import Callable
import torch, torch.nn as nn


# Define hyper parameters.
data_filename = 'testdata/tinyshakespeare.txt'
tokenizer = CharTokenizer(filename = data_filename)
max_block_size = 24
train_fraction = 0.9
batch_size = 32
num_batches_to_train = 1000
num_batches_to_evaluate = 10
num_decoder_blocks = 10
embedding_dimension = 64
num_heads = 8
head_dimension = 16
num_tokens_to_generate_during_evaluation = 100
torch.manual_seed(123)


# Generate train and validation datasets and data loaders.
train_dataset = TextDataset(max_block_size, tokenizer, 'train', train_fraction, filename = data_filename)
val_dataset = TextDataset(max_block_size, tokenizer, 'val', train_fraction, filename = data_filename)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True)


model = GPT(num_decoder_blocks, tokenizer.vocabulary_length(), embedding_dimension, num_heads, head_dimension, max_block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
evaluator = Evaluator()

for (batch_index, train_batch) in enumerate(train_dataloader):
    if batch_index > num_batches_to_train:
        print('Reached maximum number of matches. Training is now complete.')
        break

    train_features = train_batch['features']
    train_labels = train_batch['labels']
    predictions, loss = model(train_features, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 10 == 0:
        (train_loss, val_loss) = evaluator.evaluate_train_and_validation_loss(train_dataloader, val_dataloader, model, num_batches_to_evaluate)
        generated_text = evaluator.generate_text(model, num_tokens_to_generate_during_evaluation, tokenizer)
        print(f' Batch index: {batch_index}, train loss: {train_loss}, val_loss: {val_loss}, generated text\n {generated_text}')
