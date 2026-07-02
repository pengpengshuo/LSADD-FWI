import torch
import deepwave

from deepwave import scalar
from matplotlib import pyplot as plt
from _2_preliminary_setting import Preliminary,DIR

def add_gaussian_noise(data, noise_std_ratio=0.5,save_file=None): #Add noise to the observation data
    """
    Args:
        data (torch.Tensor): Shape (n_shots, n_receivers, nt), noise-free seismic data.
        noise_std_ratio (float): Noise standard deviation ratio (e.g., 0.75 means 0.75σ0).
    Returns:
        noisy_data (torch.Tensor): Data with added Gaussian noise.
    """
    std0 = data.std() #Calculate the standard deviation of the data
    noise = torch.randn_like(data) #Generate Gaussian noise with a mean of 0 and a standard deviation of 1
    noise = noise * (noise_std_ratio * std0) #Generate the required Gaussian noise
    noisy_data = data + noise #Generate observation data containing noise
    noisy_data.numpy().tofile(save_file) #Save data

def wavefield_visualization(wavefield,n_shot=20): #Visualize single shot data
    print(wavefield.min(),wavefield.max())
    vmin, vmax = torch.quantile(wavefield[0], torch.tensor([0.05, 0.95]))
    print(vmin,vmax)
    _, ax = plt.subplots(1, 1, figsize=(10.5, 7), sharey=True)
    ax.imshow(wavefield[n_shot].cpu().T, aspect='auto',
              cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    plt.close()

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    preliminary = Preliminary().to_device(device)
    observed_data = (torch.from_file(str(DIR / 'marmousi_data20.bin'),
                                     size=preliminary.n_shots * preliminary.n_receivers_per_shot * preliminary.nt)
                     .reshape(preliminary.n_shots, preliminary.n_receivers_per_shot, preliminary.nt).to(
        device))  # Load the observed data
    add_gaussian_noise(observed_data, 1)

