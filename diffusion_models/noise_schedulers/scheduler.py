# Implementation of Noise Schedulers for image diffusion.

import torch
from torch import nn
import math

class NoiseScheduler:
    '''
        Noise scheduler class. This class provides add_noise() which adds noise to input image, and
        denoise() which denoises an input image using the diffusion model.
    '''
    def __init__(self, max_timesteps, device) -> None:
        '''
            Initializer function.

            Args:
                1. max_timesteps: Maximum number of timesteps to use for adding noise and denoising.
                2. device: Device to run the computations.

            Returns:
                None
        '''
        super().__init__()
        self.max_timesteps = max_timesteps
        self.device = device

        self.beta_small = 1e-4
        self.beta_large = 0.0002

    def beta(self, t):
        return self.beta_small + (t / self.max_timesteps) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    @torch.no_grad()
    def add_noise(self, input: torch.Tensor) -> torch.Tensor:
        '''
            Add noise to the input image using the diffusion forward process.

            Args:
                1. input: Input image to add the noise to.

            Returns:
                Noisy image.
        '''
        batch_size = input.shape[0]
        time_steps = torch.randint(0, self.max_timesteps, (batch_size,), device=self.device)

        noisy_images = []
        epsilons = torch.randn(input.shape, device = self.device)
        for i in range(len(time_steps)):
            a_hat = self.alpha_bar(time_steps[i])
            noisy_images.append(
                (math.sqrt(a_hat) * input[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noisy_images = torch.stack(noisy_images, dim=0)
        return noisy_images, epsilons

    @torch.no_grad()
    def denoise(self, noisy_images: torch.Tensor, timestep: int, model: nn.Module, input_text: torch.Tensor) -> torch.Tensor:
        '''
            Denoise noisy images using the image diffusion model.

            Args:
                1. noisy_image: [Batch, 3, Width, Height] tensor containing noisy images.
                2. timestep: Number of timesteps to apply the reverse diffusion process.
                3. model: Image diffusion model.
                4. input_text: [Batch, BlockSize] tensor containing text associated with the noisy images.

            Returns:
                [Batch, 3, Width, Height] tensor containing denoised images.
        '''
        z = torch.randn(noisy_images.shape, device = self.device) if timestep > 1 else 1
        e_hat = model(noisy_images, input_text)
        pre_scale = 1 / math.sqrt(self.alpha(timestep))
        e_scale = (1 - self.alpha(timestep)) / math.sqrt(1 - self.alpha_bar(timestep))
        post_sigma = math.sqrt(self.beta(timestep)) * z
        denoised_images = pre_scale * (noisy_images - e_scale * e_hat) + post_sigma
        return denoised_images
