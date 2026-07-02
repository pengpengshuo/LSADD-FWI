from itertools import product

import torch
from torch import nn
from scipy.ndimage import gaussian_filter
from _2_preliminary_setting import Preliminary,Filterd_freq_Preliminary,DIR
from _3_model import CNN_FWI,LSADD_FWI,LSADD_FWI_temp
from _0_visualization_tool import loss_visualization,velocity_visualization

preliminary=Preliminary() #Define the relevant settings for forward modeling
v_true=preliminary.forward(str(DIR/"marmousi_vp20.bin"),'cpu') #Load the true velocity model
v_init_smooth = (torch.tensor(1 / gaussian_filter(1 / v_true.cpu().numpy(), 15))) #Set smooth initial model
v_init_linear=torch.linspace(2500,4500,144).unsqueeze(0).expand(432,-1) #Set linear initial model

observed_data=(torch.from_file(str(DIR/'marmousi_data20.bin'),size=preliminary.n_shots*preliminary.n_receivers_per_shot*preliminary.nt)
               .reshape(preliminary.n_shots,preliminary.n_receivers_per_shot,preliminary.nt))  #Load the observed data

opti_list=["smooth_CNN","smooth_LSADD","linear_CNN","linear_LSADD","linear_LSADD_noise","linear_LSADD_filterd","constant_LSADD"]
i=3
result="result"
if opti_list[i]=="smooth_CNN":
    input_CNN=(observed_data[::5,68:357,:864]).permute(0,2,1).contiguous().unsqueeze(0) #load input data of the neural network
    lr_list = [0]
    for lr in lr_list:
        model = CNN_FWI() #Generate reparameterized network model
        file = DIR / f'{result}/smooth/CNN_FWI/train_epoch_2500th.pth' #Directory address of training results
        print(f'{lr}')
        loss_visualization(file) #Visual loss function
        velocity_visualization(input_CNN, model, v_init_smooth, v_true, file,
                               save=DIR / f'{result}/smooth/CNN_FWI/train_epoch_2500th') #Visual velocity model

if opti_list[i]=="smooth_LSADD":
    input_LSADD = (observed_data[::5, 19:403:2, :864]).permute(0, 2, 1).contiguous().unsqueeze(0)
    lr_list = [0]
    for lr in lr_list:
        model = LSADD_FWI()
        file = DIR / f'{result}/smooth/LSADD_FWI/train_epoch_2500th.pth'
        print(f'{lr}')
        loss_visualization(file)
        velocity_visualization(input_LSADD, model, v_init_smooth, v_true, file, save=DIR / f'{result}/smooth/LSADD_FWI/train_epoch_2500th')

if opti_list[i]=="linear_CNN":
    velocity_data = (observed_data[..., 1:] - observed_data[..., :-1]) / preliminary.dt
    input_CNN_velocity = (velocity_data[::5, 68:357, :864]).permute(0, 2, 1).contiguous().unsqueeze(0)
    lr_list = [0]
    for lr in lr_list:
        model = CNN_FWI()
        file = DIR / f'{result}/linear/CNN_FWI/train_epoch_2500th.pth'
        print(f'{lr}')
        loss_visualization(file)
        velocity_visualization(input_CNN_velocity, model, v_init_linear, v_true, file,
                               save=DIR / f'{result}/linear/CNN_FWI/train_epoch_2500th')

if opti_list[i]=="linear_LSADD":
    input_LSADD = (observed_data[::5, 19:403:2, :864]).permute(0, 2, 1).contiguous().unsqueeze(0)
    lr_list = [0]
    for lr in lr_list:
        model = LSADD_FWI()
        file = DIR / f'{result}/linear/LSADD_FWI/train_epoch_2500th.pth'
        print(f'{lr}')
        loss_visualization(file)
        velocity_visualization(input_LSADD, model, v_init_linear, v_true, file,
                               save= DIR / f'{result}/linear/LSADD_FWI/train_epoch_2500th')

if opti_list[i]=="linear_LSADD_noise":
    input_LSADD = (observed_data[::5, 19:403:2, :864]).permute(0, 2, 1).contiguous().unsqueeze(0)
    noise_list = [1]
    for noise in noise_list:
        observed_data_noise = (torch.from_file(str(DIR / f'marmousi_data20_noise_{noise}.bin'),
                                               size=preliminary.n_shots * preliminary.n_receivers_per_shot * preliminary.nt)
        .reshape(preliminary.n_shots, preliminary.n_receivers_per_shot, preliminary.nt))
        lr_list=[0]
        for lr in lr_list:
            model = LSADD_FWI()
            file = DIR / f'{result}/linear/LSADD_FWI_noise_{noise}/{lr}/train_epoch_2500th.pth'
            print(f'{lr}')
            loss_visualization(file)
            velocity_visualization(input_LSADD, model, v_init_linear, v_true, file,
                                   save=DIR / f'{result}/linear/LSADD_FWI_noise_{noise}/{lr}/train_epoch_2500th')

if opti_list[i]=="linear_LSADD_filterd":
    cutoff_list = [4, 6]
    for cutoff in cutoff_list:
        filterd_freq_preliminary = Filterd_freq_Preliminary(cutoff=cutoff)
        filterd_freq_obsrved_data = (torch.from_file(str(DIR / f'marmousi_data20_filterd_{cutoff}.bin'),
                                                     size=filterd_freq_preliminary.n_shots * filterd_freq_preliminary.n_receivers_per_shot * filterd_freq_preliminary.nt)
                                     .reshape(filterd_freq_preliminary.n_shots,
                                              filterd_freq_preliminary.n_receivers_per_shot,
                                              filterd_freq_preliminary.nt))
        filterd_freq_input_LSADD = (filterd_freq_obsrved_data[::5, 19:403:2, :864]).permute(0, 2,
                                                                                           1).contiguous().unsqueeze(0)
        lr_list = [0]
        for lr in lr_list:
            model = LSADD_FWI()
            file = DIR / f'{result}/linear/LSADD_FWI_filterd_{cutoff}/train_epoch_2500th.pth'
            print(f'cutoff:{cutoff};lr:{lr}')
            loss_visualization(file)
            velocity_visualization(filterd_freq_input_LSADD, model, v_init_linear, v_true, file,
                                save=DIR / f'{result}/linear/LSADD_FWI_filterd_{cutoff}/train_epoch_2500th')

if opti_list[i]=="constant_LSADD":
    input_LSADD = (observed_data[::5, 19:403:2, :864]).permute(0, 2, 1).contiguous().unsqueeze(0)
    v_init_constant = (torch.ones(1, 144) * 3500).expand(432, -1)
    lr_list = [0]
    config_list = [
        {"temp": 10},
    ]
    for config, lr in product(config_list, lr_list):
        model = LSADD_FWI_temp(**config)
        file = DIR / f'{result}/constant/train_epoch_2500th.pth'
        print(f'{lr}')
        loss_visualization(file)
        velocity_visualization(input_LSADD, model, v_init_constant, v_true, file,
                               save=DIR / f'{result}/constant/train_epoch_2500th')
