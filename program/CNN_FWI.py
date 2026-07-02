import os
import torch
import time
from scipy.ndimage import gaussian_filter
from _2_preliminary_setting import Preliminary,DIR,Filterd_freq_Preliminary
from _4_training_function import set_seed,batch_train_one_step
from _0_learning_rate_adjust import CosineWarmupScheduler
from _3_model import CNN_FWI


def CNN_FWI_inversion_smooth(preliminary,input_CNN,v_true,observed_data,device):
    set_seed() #Set random seeds
    v_init_smooth = (torch.tensor(1 / gaussian_filter(1 / v_true.cpu().numpy(), 15))).to(device) #Set initial model
    lr_list=[0.008,0.01] #Set learning rate
    for lr in lr_list:
        model=CNN_FWI().to(device) #Generate reparameterized network model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Generate Adam optimizer
        loss_history = [] #Define an empty list to store the loss during training
        start_epoch = [0]
        total_epoch = 2500 #Total number of training
        os.makedirs(DIR / f'run_result/smooth/CNN_FWI/{lr}', exist_ok=True)
        file = DIR / f'run_result/smooth/CNN_FWI/{lr}' #Create a directory to store training results
        n_shots = preliminary.n_shots  # 50
        batch_size = 25
        total_batch = n_shots // batch_size
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch*total_batch,
                                          min_lr=0.0006) #Define learning rate adjustment strategy
        for epoch in range(start_epoch[0], total_epoch):
            start_time = time.time()
            for batch_idx in range(total_batch):
                batch_train_one_step(model, optimizer, scheduler, input_CNN, v_init_smooth, v_true, batch_idx, total_batch,
                                     observed_data, preliminary,
                                     loss_history, epoch, file, start_time, transform=True, save_num=2500) #Training steps

def CNN_FWI_inversion_linear(preliminary,input_CNN_velocity,v_true,observed_data,device):
    set_seed()
    v_init_linear = torch.linspace(2500, 4500, 144).unsqueeze(0).expand(432, -1).to(device)
    lr_list=[0.011,0.013,0.015,0.017,0.019]
    for lr in lr_list:
        model=CNN_FWI().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_history = []
        start_epoch = [0]
        total_epoch = 2500
        os.makedirs(DIR / f'run_result/linear/CNN_FWI/{lr}', exist_ok=True)
        file = DIR / f'run_result/linear/CNN_FWI/{lr}'
        n_shots = preliminary.n_shots  # 50
        batch_size = 25
        total_batch = n_shots // batch_size  # 5
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch*total_batch, min_lr=0.0006)
        for epoch in range(start_epoch[0], total_epoch):
            start_time = time.time()
            for batch_idx in range(total_batch):
                batch_train_one_step(model, optimizer, scheduler, input_CNN_velocity, v_init_linear, v_true, batch_idx, total_batch,
                                     observed_data, preliminary,
                                     loss_history, epoch, file, start_time, transform=True, save_num=2500)

def CNN_FWI_noise_inversion(preliminary,input_CNN_velocity,v_true,device):
    set_seed()
    v_init_linear = torch.linspace(2500, 4500, 144).unsqueeze(0).expand(432, -1).to(device)
    noise_list = [0.5,1]
    for noise in noise_list:
        observed_data_noise = (torch.from_file(str(DIR/f'marmousi_data20_noise_{noise}.bin'),
                                               size=preliminary.n_shots * preliminary.n_receivers_per_shot * preliminary.nt)
        .reshape(preliminary.n_shots, preliminary.n_receivers_per_shot, preliminary.nt).to(
            device))
        lr_list = [0.005, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02]
        for lr in lr_list:
            model = CNN_FWI().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_history = []
            start_epoch = [0]
            total_epoch = 2500
            os.makedirs(DIR/f'run_result/linear/CNN_FWI_noise_{noise}/{lr}', exist_ok=True)
            file = DIR/f'run_result/linear/CNN_FWI_noise_{noise}/{lr}'
            n_shots = preliminary.n_shots  # 50
            batch_size = 25
            total_batch = n_shots // batch_size  # 5
            scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch * total_batch,
                                              min_lr=0.0006)
            for epoch in range(start_epoch[0], total_epoch):
                start_time = time.time()
                for batch_idx in range(total_batch):
                    batch_train_one_step(model, optimizer, scheduler, input_CNN_velocity, v_init_linear, v_true,
                                         batch_idx,
                                         total_batch, observed_data_noise, preliminary,
                                         loss_history, epoch, file, start_time, transform=True, save_num=2500)

def filterd_freq_CNN_FWI_inversion(v_true,device):
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
        velocity_filterd_freq_data = (filterd_freq_obsrved_data[..., 1:] - filterd_freq_obsrved_data[..., :-1]) / filterd_freq_preliminary.dt
        input_CNN_velocity_filterd_freq=(velocity_filterd_freq_data[::5,68:357,:864]).permute(0,2,1).contiguous().unsqueeze(0)
        lr_list = [0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
        for lr in lr_list:
            model = CNN_FWI().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_history = []
            start_epoch = [0]
            total_epoch = 2500
            os.makedirs(DIR / f'run_result/linear/CNN_FWI_filterd_{cutoff}/{lr}', exist_ok=True)
            file = DIR / f'run_result/linear/CNN_FWI_filterd_{cutoff}/{lr}'
            n_shots = filterd_freq_preliminary.n_shots  # 50
            batch_size = 25
            total_batch = n_shots // batch_size
            scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=20, total_epochs=total_epoch * total_batch,
                                              min_lr=0.0006)
            for epoch in range(start_epoch[0], total_epoch):
                start_time = time.time()
                for batch_idx in range(total_batch):
                    batch_train_one_step(model, optimizer, scheduler, input_CNN_velocity_filterd_freq, v_init_linear, v_true,
                                         batch_idx,
                                         total_batch, filterd_freq_obsrved_data, filterd_freq_preliminary,
                                         loss_history, epoch, file, start_time, transform=True, save_num=2500)


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
velocity_data = (observed_data[..., 1:] - observed_data[..., :-1]) / preliminary.dt # the particle-velocity wavefield
input_CNN=(observed_data[::5,68:357,:864]).permute(0,2,1).contiguous().unsqueeze(0) #Load the input wavefield data of the neural network
input_CNN_velocity=(velocity_data[::5,68:357,:864]).permute(0,2,1).contiguous().unsqueeze(0)#Load the input velocity data of the neural network

'''According to the requirements, choose one of the following functions to run'''
CNN_FWI_inversion_smooth(preliminary,input_CNN,v_true,observed_data,device) #Inversion of smooth initial model
CNN_FWI_inversion_linear(preliminary,input_CNN_velocity,v_true,observed_data,device) #Inversion of linear initial model
CNN_FWI_noise_inversion(preliminary,input_CNN_velocity,v_true,device) #Inversion of observation data containing noise
filterd_freq_CNN_FWI_inversion(v_true,device) #Inversion of observation data lacking low-frequency information
