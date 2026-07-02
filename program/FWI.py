import os
import torch
from scipy.ndimage import gaussian_filter
from _2_preliminary_setting import Preliminary,DIR,Filterd_freq_Preliminary
from _4_training_function import set_seed,convention_train_one_step
from _0_learning_rate_adjust import CosineWarmupScheduler
from _3_model import FWI

def FWI_inversion_smooth(preliminary,v_true,observed_data,device):
    set_seed() #Set random seeds
    v_init_smooth = (torch.tensor(1 / gaussian_filter(1 / v_true.cpu().numpy(), 15))).to(device) #Set initial model
    lr_list = [0.1,0.3,0.5,0.7,0.9] #Set learning rate
    for lr in lr_list:
        model = FWI(v_init_smooth, v_true.min(), v_true.max()).to(device) #Generate inversion velocity model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Generate Adam optimizer
        loss_history = [] #Define an empty list to store the loss during training
        start_epoch = [0]
        total_epoch = 2100 #Total number of training
        os.makedirs(DIR / f'run_result/smooth/FWI/{lr}', exist_ok=True)
        file = DIR / f'run_result/smooth/FWI/{lr}' #Create a directory to store training results
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch,
                                          min_lr=0.0006) #Define learning rate adjustment strategy
        for epoch in range(start_epoch[0], total_epoch):
            convention_train_one_step(model, optimizer, scheduler,
                                      observed_data, preliminary, 1e-3, 1,
                                      loss_history, epoch, file, save_num=2100) #Training steps

def FWI_inversion_linear(preliminary,v_true,observed_data,device):
    set_seed()
    v_init_linear = torch.linspace(2500, 4500, 144).unsqueeze(0).expand(432, -1).to(device)
    lr_list = [0.1,0.3,0.5,0.7,0.9]
    for lr in lr_list:
        model = FWI(v_init_linear, v_true.min(), v_true.max()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_history = []
        start_epoch = [0]
        total_epoch = 2100
        os.makedirs(DIR / f'run_result/linear/FWI/{lr}', exist_ok=True)
        file = DIR / f'run_result/linear/FWI/{lr}'
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch, min_lr=0.0006)
        for epoch in range(start_epoch[0], total_epoch):
            convention_train_one_step(model, optimizer, scheduler,
                                      observed_data, preliminary, 1e-3, 1,
                                      loss_history, epoch, file, save_num=2100)

def FWI_noise_inversion(preliminary,v_true,device):
    set_seed()
    v_init_linear = torch.linspace(2500, 4500, 144).unsqueeze(0).expand(432, -1).to(device)
    noise_list = [0.5, 1]
    for noise in noise_list:
        observed_data_noise = (torch.from_file(str(DIR/f'marmousi_data20_noise_{noise}.bin'),
                                               size=preliminary.n_shots * preliminary.n_receivers_per_shot * preliminary.nt)
        .reshape(preliminary.n_shots, preliminary.n_receivers_per_shot, preliminary.nt).to(
            device))
        lr_list = [0.1,0.3,0.5,0.7,0.9]
        for lr in lr_list:
            model = FWI(v_init_linear, v_true.min(), v_true.max()).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_history = []
            start_epoch = [0]
            total_epoch = 2100
            os.makedirs(DIR / f'run_result/linear/FWI_noise_{noise}/{lr}', exist_ok=True)
            file = DIR / f'run_result/linear/FWI_noise_{noise}/{lr}'
            scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch,
                                              min_lr=0.0006)
            for epoch in range(start_epoch[0], total_epoch):
                convention_train_one_step(model, optimizer, scheduler,
                                          observed_data_noise, preliminary, 1e-3, 1,
                                          loss_history, epoch, file, save_num=2100)

def filterd_freq_FWI_inversion(v_true,device):
    set_seed()
    v_init_linear = torch.linspace(2500, 4500, 144).unsqueeze(0).expand(432, -1).to(device)
    cutoff_list=[4,6]
    for cutoff in cutoff_list:
        filterd_freq_preliminary = Filterd_freq_Preliminary(cutoff=cutoff).to_device(device)
        filterd_freq_obsrved_data = (torch.from_file(str(DIR/f'marmousi_data20_filterd_{cutoff}.bin'),
                                                     size=filterd_freq_preliminary.n_shots * filterd_freq_preliminary.n_receivers_per_shot * filterd_freq_preliminary.nt)
                                     .reshape(filterd_freq_preliminary.n_shots,
                                              filterd_freq_preliminary.n_receivers_per_shot,
                                              filterd_freq_preliminary.nt).to(device))
        lr_list = [0.1,0.3,0.5,0.7,0.9]
        for lr in lr_list:
            model = FWI(v_init_linear, v_true.min(), v_true.max()).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_history = []
            start_epoch = [0]
            total_epoch = 2100
            os.makedirs(DIR / f'run_result/linear/FWI_filterd_{cutoff}/{lr}', exist_ok=True)
            file = DIR / f'run_result/linear/FWI_filterd_{cutoff}/{lr}'
            scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch,
                                              min_lr=0.0006)
            for epoch in range(start_epoch[0], total_epoch):
                convention_train_one_step(model, optimizer, scheduler,
                                          filterd_freq_obsrved_data, filterd_freq_preliminary, 1e-3, 1,
                                          loss_history, epoch, file, save_num=2100)

device_num=0
if torch.cuda.is_available():
    torch.cuda.set_device(device_num)
    device = torch.device("cuda")
    print(f"Training {device_num+1} using device: {torch.cuda.current_device()}")
else:
    device="cpu"

preliminary=Preliminary().to_device(device) #Define the relevant settings for forward modeling

v_true=preliminary.forward(str(DIR/"marmousi_vp20.bin"),device) #Load the true velocity model

observed_data=(torch.from_file(str(DIR/'marmousi_data20.bin'),size=preliminary.n_shots*preliminary.n_receivers_per_shot*preliminary.nt)
               .reshape(preliminary.n_shots,preliminary.n_receivers_per_shot,preliminary.nt).to(device)) #Load the observed data
'''According to the requirements, choose one of the following functions to run'''
FWI_inversion_smooth(preliminary,v_true, observed_data, device) #Inversion of smooth initial model
FWI_inversion_linear(preliminary,v_true, observed_data, device) #Inversion of linear initial model
FWI_noise_inversion(preliminary,v_true, device) #Inversion of observation data containing noise
filterd_freq_FWI_inversion(v_true, device) #Inversion of observation data lacking low-frequency information