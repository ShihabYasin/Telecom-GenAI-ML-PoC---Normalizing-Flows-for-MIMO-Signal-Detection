import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform

# Generate simulated MIMO data
def generate_mimo_data(num_samples, num_antennas, snr_db=20):
    # Generate random QPSK symbols
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=(num_samples, num_antennas))
    symbols_real = np.real(symbols)
    symbols_imag = np.imag(symbols)
    symbols = np.stack([symbols_real, symbols_imag], axis=-1)
    
    # Add Gaussian noise
    snr = 10**(snr_db/10)
    noise_std = 1/np.sqrt(2*snr)  # Complex noise variance
    noise_real = np.random.normal(0, noise_std, (num_samples, num_antennas))
    noise_imag = np.random.normal(0, noise_std, (num_samples, num_antennas))
    noisy_symbols = symbols + np.stack([noise_real, noise_imag], axis=-1)
    
    return torch.tensor(symbols, dtype=torch.float32), torch.tensor(noisy_symbols, dtype=torch.float32)

# Normalizing Flow model for noise modeling
class NoiseModel(nn.Module):
    def __init__(self, input_dim):
        super(NoiseModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim*2, 64),  # 2 for real/imag
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim*2)  # Output loc and scale for each dimension
        )
    
    def forward(self, x):
        return self.net(x)

# Create normalizing flow transformation
def create_flow(noise_model, input_dim):
    base_dist = Normal(torch.zeros(input_dim*2), torch.ones(input_dim*2))
    
    # Define transform using the noise model
    def transform(x):
        params = noise_model(x)
        loc = params[..., :input_dim*2:2]
        scale = torch.exp(params[..., 1:input_dim*2:2])
        return x * scale + loc
    
    return TransformedDistribution(base_dist, [AffineTransform(loc=0, scale=1), transform])

# Training function
def train_noise_model(noise_model, clean_data, noisy_data, epochs=200, lr=0.001):
    optimizer = optim.Adam(noise_model.parameters(), lr=lr)
    input_dim = clean_data.shape[1]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Create flow distribution
        flow = create_flow(noise_model, input_dim)
        
        # Compute negative log likelihood
        nll = -flow.log_prob(noisy_data.view(-1, input_dim*2)).mean()
        
        # Backpropagation
        nll.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], NLL: {nll.item():.4f}')
    
    return noise_model

# Denoising function
def denoise_signal(noise_model, noisy_data):
    input_dim = noisy_data.shape[1]
    flow = create_flow(noise_model, input_dim)
    
    # For denoising, we can use the mean of the distribution
    with torch.no_grad():
        denoised = flow.transforms[-1](noisy_data.view(-1, input_dim*2))
    
    return denoised.view_as(noisy_data)

# Main execution
if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    num_antennas = 4  # 4x4 MIMO system
    snr_db = 15
    
    # Generate data
    clean_data, noisy_data = generate_mimo_data(num_samples, num_antennas, snr_db)
    
    # Initialize model
    model = NoiseModel(num_antennas)
    
    # Train model
    model = train_noise_model(model, clean_data, noisy_data)
    
    # Denoise a sample
    denoised = denoise_signal(model, noisy_data[:10])
    
    print("Noise modeling completed!")
    print("Original clean:", clean_data[0])
    print("Noisy input:", noisy_data[0])
    print("Denoised output:", denoised[0])
