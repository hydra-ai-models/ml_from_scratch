# Trainer script for training image diffusion models.
# Move to the diffusion_models directory. Now, run this script using
#   python -m trainer.trainer_diffusion_models

from PIL import Image
from datasets import load_dataset
from dataset_utils.dataset_utils import verify_columns, visualize_dataset, preprocess_dataset, collate_fn
from layers.unet import UNetWithCrossAttention
from noise_schedulers.scheduler import NoiseScheduler
from evaluator.evaluate_utils import visualize_predictions
import torch
from transformers import CLIPTokenizer, tokenization_utils_base
from typing import Union, Optional, Any
from functools import partial
from diffusers import AutoencoderKL
from dataclasses import dataclass
from torch import nn
import math


# Hyperparameters
# Dataset hyper parameters.
dataset_name = 'lambdalabs/pokemon-blip-captions'
tokenizer_name = 'CompVis/stable-diffusion-v1-4'
text_encoder_name = 'CompVis/stable-diffusion-v1-4'
image_encoder_name = 'CompVis/stable-diffusion-v1-4'
train_split_name = 'train'
image_column = 'image'
caption_column = 'text'

# Architecture hyperparamters.
down_sizes = [256, 128, 64]
up_sizes = [64, 128, 256]

# Training hyper parameters.
seed = 123
batch_size = 4
num_epochs = 1
num_batches = 40
num_batches_to_visualize = 2
max_time_steps = 100

# Setting device.
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        ValueError("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        ValueError("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

device = torch.device('mps')

# Load dataset, verify columns and dataloader.
torch.manual_seed(seed)
dataset = load_dataset(dataset_name)
assert train_split_name in dataset.keys(), f'{train_split_name} is not a valid key in the dataset dictionary.'
train_dataset = dataset[train_split_name]
verify_columns(train_dataset, image_column, caption_column)
visualize_dataset(train_dataset, image_column, caption_column, False)

tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name, subfolder="tokenizer")
preprocess_dataset_partial = partial(preprocess_dataset, image_size = 128, image_column = image_column, caption_column = caption_column, tokenizer = tokenizer)
train_dataset.set_transform(preprocess_dataset_partial)
visualize_dataset(train_dataset, image_column, caption_column, tokenizer, False)

collate_fn_partial = partial(collate_fn, tokenizer = tokenizer)
train_dataloader = torch.utils.data.DataLoader(dataset['train'], shuffle = True, batch_size = batch_size, collate_fn = collate_fn_partial)

# Define model and optimizer.
model = UNetWithCrossAttention(skip_connections = True)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-3)
noise_scheduler = NoiseScheduler(max_time_steps, device)

# Train model.
for epoch_index in range(num_epochs):
    for batch_index, batch in enumerate(train_dataloader):
        if batch_index >= num_batches:
            break
        input_image = batch['image'].to(device)
        noisy_image, epsilons = noise_scheduler.add_noise(input_image)
        input_text = batch['text'].to(device)
        noise_predictions = model(noisy_image, input_text)
        optimizer.zero_grad()
        loss = nn.MSELoss()(epsilons, noise_predictions)
        loss.backward()
        optimizer.step()
        if batch_index % num_batches_to_visualize == 0:
            print(f'Epoch index : {epoch_index}. Batch index: {batch_index}. Loss: {loss}.')
            denoised_image = noisy_image
            for time_step in range(max_time_steps, 0, -1):
                denoised_image = noise_scheduler.denoise(denoised_image, time_step, model, input_text)
            visualize_predictions(input_image, input_text, noisy_image, denoised_image)
