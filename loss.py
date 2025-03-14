import torch
import torch.nn as nn
import numpy as np
from noise import snoise3

def generate_simplex_noise(shape, scale=10.0):
    """Generate Simplex noise with given shape."""
    B, C, H, W = shape
    simplex_noise = torch.zeros(shape, device="cuda")  # Assuming GPU usage

    for b in range(B):
        for h in range(H):
            for w in range(W):
                simplex_noise[b, 0, h, w] = snoise3(h / scale, w / scale, b / scale)

    # Normalize to match Gaussian distribution properties
    simplex_noise = (simplex_noise - simplex_noise.mean()) / simplex_noise.std()
    
    return simplex_noise

def get_loss(model, x_0, t, config):
    print("Loss ftnnnnnnn")
    x_0 = x_0.to(config.model.device)
    
    betas = torch.linspace(
        config.model.beta_start, 
        config.model.beta_end, 
        config.model.trajectory_steps, 
        dtype=torch.float, 
        device=config.model.device
    )
    
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    at = alpha_bar.index_select(0, t).view(-1, 1, 1, 1)

    # Replace Gaussian noise with Simplex noise
    e = generate_simplex_noise(x_0.shape).to(x_0.device)

    print("e shape", e.shape)
    print("x_0 shape", x_0.shape)
    # e = (e - e.mean()) / e.std()

    # Apply forward diffusion process
    x_t = at.sqrt() * x_0 + (1 - at).sqrt() * e 

    # Predict noise using the model
    output = model(x_t, t.float())
    loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    print(f'Loss: {loss}')
    # Compute denoising loss (Normalized MSE)
    return loss


# def get_loss(model, x_0, t, config):
#     x_0 = x_0.to(config.model.device)
#     betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
#     b = torch.tensor(betas).type(torch.float).to(config.model.device)
#     e = torch.randn_like(x_0, device = x_0.device)
#     at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)


#     x = at.sqrt() * x_0 + (1- at).sqrt() * e 
#     output = model(x, t.float())
#     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)