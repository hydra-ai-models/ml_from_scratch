# Trainer script for GPT pretraining using HuggingFace accelerate for largescale
# distributed training.
# Move to the large_language_models directory. Then create accelerate config by running
#   accelerate config
#
# See README file for details on options to set for the config command. Now run training using
#   accelerate launch -m trainer.trainer_gpt_pretraining_with_accelerate

import os

from datasets.text_dataset import TextDataset
from tokenizers.char_tokenizer import CharTokenizer
from layers.gpt import GPT
from layers import layer_utils
from evaluate import Evaluator

from collections.abc import Callable
import torch, torch.nn as nn

def train():
    # Main function for launching the training job.

    # Define hyperparameters.

    # Dataset hyperparameters.
    data_filename = 'testdata/tinyshakespeare.txt'
    train_fraction = 0.9

    # Tokenizer hyperparameters.
    tokenizer = CharTokenizer(filename = data_filename)

    # Architecture hyperparameters.
    embedding_dimension = 64
    num_heads = 8
    head_dimension = 16
    num_decoder_blocks = 10

    # Training hyperparameters.
    max_block_size = 24
    batch_size = 32
    num_batches_to_train = 200

    # Evaluation hyperparameters.
    num_batches_to_evaluate = 10
    num_tokens_to_generate_during_evaluation = 10
    num_batches_between_evaluations = 10

    # Output parameters.
    output_model_path = 'output/gpt_pretrained_model.pt'
    output_params_path = 'output/num_parameters_gpt_pretraining.yaml'

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
    model = GPT(num_decoder_blocks, tokenizer.vocabulary_length(), embedding_dimension, num_heads, head_dimension, max_block_size)
    num_model_parameters = layer_utils.num_parameters(model, output_params_path)
    print(f'Number of parameters in the model is {num_model_parameters["total_trainable_parameters"]}.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    evaluator = Evaluator()

    # Defining HF accelerator.
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f'Running training and evaluation on device {device}.')

    (model, optimizer, train_dataloader, scheduler) = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    model.to(device)

    # Perform model training and evaluation.
    for (batch_index, train_batch) in enumerate(train_dataloader):
        if batch_index > num_batches_to_train:
            print('Reached maximum number of matches. Training is now complete.')
            torch.save(model.state_dict(), output_model_path)
            break

        train_features = train_batch['features'].to(device)
        train_labels = train_batch['labels'].to(device)
        predictions, loss = model(train_features, train_labels)

        # Note that making the gradients zero is not needed with the accelerator, as it will take care of this.
        #optimizer.zero_grad()
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        if batch_index % num_batches_between_evaluations == 0:
            (train_loss, val_loss) = evaluator.evaluate_train_and_validation_loss(train_dataloader, val_dataloader, model, num_batches_to_evaluate, device)
            generated_text = evaluator.generate_text(model, num_tokens_to_generate_during_evaluation, tokenizer, device)
            print(f' Batch index: {batch_index}, train loss: {train_loss}, val_loss: {val_loss}, generated text\n {generated_text}')

if __name__ == '__main__':
    train()
