import torch
import deepwave
from deepwave import scalar
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.fft
from _0_signal_processing import highpass_filter_fft_controlled
from _0_directory_setting import DIR
'''
Setup of velocity model, source points, source wavelet, receivers, and observed data for the experiment:
1.Source coordinates: self.source_locations
2.Receiver coordinates: self.receiver_locations
3.Source wavelet (amplitudes): self.source_amplitudes
4.Generation of observed data
'''
class Preliminary(): #Standard data settings
    def __init__(self,dx=20.0,ny_data=432,nx_data=144,
                 n_shots=50, n_sources_per_shot=1, d_source=5, first_source=90, source_depth=2,
                 n_receivers_per_shot=424, d_receiver=1, first_receiver=4, receiver_depth=2,
                 freq=10,nt=1000,dt=0.006
                 ):
        self.dx=dx # Spatial grid spacing
        self.ny_data=ny_data  # Number of horizontal grid points
        self.nx_data=nx_data  # Number of vertical grid points
        self.dt=dt  # Time sampling interval
        self.freq=freq # Peak frequency of the Ricker wavelet
        self.n_shots=n_shots  # Number of shot gathers
        self.n_receivers_per_shot=n_receivers_per_shot # Number of receivers per shot
        self.nt=nt # Number of time samples
        #Define the location of the sources
        self.source_locations=torch.zeros(n_shots,n_sources_per_shot,2,
                                          dtype=torch.long)
        self.source_locations[...,1]=source_depth
        self.source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                             first_source)
        # Define the location of the receivers
        self.receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                         dtype=torch.long)
        self.receiver_locations[..., 1] = receiver_depth
        self.receiver_locations[:, :, 0] = (
            (torch.arange(n_receivers_per_shot) * d_receiver +
             first_receiver)
            .repeat(n_shots, 1)
        )
        # Creating source wavelets
        peak_time=1.5 / freq
        self.source_amplitudes = (
            deepwave.wavelets.ricker(freq, nt, dt, peak_time)
            .repeat(n_shots, n_sources_per_shot, 1)
            )

    def to_device(self,device):
        self.source_locations=self.source_locations.to(device)
        self.source_amplitudes=self.source_amplitudes.to(device)
        self.receiver_locations=self.receiver_locations.to(device)
        return self

    def forward(self,file,device):
        # Load the original velocity model
        v=torch.from_file(file,size=self.ny_data*self.nx_data).reshape(self.ny_data,self.nx_data)
        return v.to(device)

class Filterd_freq_Preliminary(): #Filter the low-frequency components in the source wavelet
    def __init__(self,dx=20.0,ny_data=432,nx_data=144,
                 n_shots=50, n_sources_per_shot=1, d_source=5, first_source=90, source_depth=2,
                 n_receivers_per_shot=424, d_receiver=1, first_receiver=4, receiver_depth=2,
                 freq=10,nt=1000,dt=0.006,cutoff=5
                 ):
        self.dx=dx
        self.ny_data=ny_data
        self.nx_data=nx_data
        self.dt=dt
        self.freq=freq
        self.n_shots=n_shots
        self.n_receivers_per_shot=n_receivers_per_shot
        self.nt=nt
        self.source_locations=torch.zeros(n_shots,n_sources_per_shot,2,
                                          dtype=torch.long)
        self.source_locations[...,1]=source_depth
        self.source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                             first_source)
        self.receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                         dtype=torch.long)
        self.receiver_locations[..., 1] = receiver_depth
        self.receiver_locations[:, :, 0] = (
            (torch.arange(n_receivers_per_shot) * d_receiver +
             first_receiver)
            .repeat(n_shots, 1)
        )
        # Create low-frequency filtered source wavelets
        peak_time=1.5 / freq
        self.cutoff=cutoff #Cut-off frequency
        self.source_amplitudes = (
            highpass_filter_fft_controlled(deepwave.wavelets.ricker(freq, nt, dt, peak_time),dt,cutoff)
            .repeat(n_shots, n_sources_per_shot, 1)
            )

    def to_device(self,device):
        self.source_locations=self.source_locations.to(device)
        self.source_amplitudes=self.source_amplitudes.to(device)
        self.receiver_locations=self.receiver_locations.to(device)
        return self
    def forward(self,file,device):
        #Load the original velocity model
        v=torch.from_file(file,size=self.ny_data*self.nx_data).reshape(self.ny_data,self.nx_data)
        return v.to(device)

def observed_wavefield(preliminary,device,file1='marmousi_vp20.bin',file2='marmousi_data20.bin'):
    v=preliminary.forward(file1,device) #Loading velocity model

    out = scalar(v, preliminary.dx, preliminary.dt,
                 source_amplitudes=preliminary.source_amplitudes,
                 source_locations=preliminary.source_locations,
                 receiver_locations=preliminary.receiver_locations,
                 accuracy=8,
                 pml_freq=preliminary.freq
                 ) #Forward modeling of wave equation
    observed_data=out[-1] #Obtain observed data
    observed_data.cpu().numpy().tofile(file2) #save observed data

    plt.figure(figsize=(13, 4)) #Visualize the corresponding velocity model
    plt.imshow(v.cpu().T, aspect='auto', cmap='jet')
    plt.axis('off')
    plt.colorbar(label='Velocity (m/s)')
    plt.show()
    plt.close()

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    preliminary=Preliminary().to_device(device) #Define preliminary
    observed_wavefield(preliminary,device,file1="marmousi_vp20.bin",file2="marmousi_data20.bin") #Create observed data
