import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colorbar import make_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim #局部SSIM


def loss_visualization(file):
    '''
    file:Directory address of training results
    '''
    checkpoint = torch.load(file, map_location='cpu') #Load checkpoint data
    loss_history = checkpoint['loss_history'] #Obtain the loss during the training process
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=1.5)
    plt.yscale('log')
    plt.ylim(min(100,min(loss_history)*0.9), max(loss_history) * 1.1)
    max_epoch = len(loss_history)
    x_ticks = np.arange(0, max_epoch + 1, max(1, max_epoch // 10))
    plt.xticks(x_ticks)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss Curve (Log Scale)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()
    plt.close()

def velocity_visualization(input,model,v_init,v_true,file,save=False):
    '''
    input:Network input    model:Reparameterized network model   v_init:initial velocity model
    file:Directory address of training results   save:Do you want to save the visualized results
    '''
    input=input.to('cpu') #Load input data
    checkpoint=torch.load(file,map_location='cpu') #Load checkpoint data
    model.load_state_dict(checkpoint['model_state']) #Assign network weights to neural networks
    output = model(input).squeeze() #the output of the neural network
    v = (torch.nn.functional.softplus(output + v_init-v_true.min()) + v_true.min()).detach() #the velocity model
    v_np = v.numpy()
    v_true_np = v_true.numpy()
    vmax = np.max([v_np.max(), v_true_np.max()])
    vmin = np.min([v_np.min(), v_true_np.min()])


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
    })
    z = np.arange(v_true.shape[1]) * 0.02
    x = np.arange(v_true.shape[0]) * 0.02


    fig, ax = plt.subplots(1, 2, figsize=(7.5, 1.25), constrained_layout=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.87,
        top=0.96,
        bottom=0.08,
        wspace=0.03,
        hspace=0.28
    )
    positions = [ax.get_position() for ax in ax.flatten()]
    y0 = min([p.y0 for p in positions])
    y1 = max([p.y1 for p in positions])
    height = y1 - y0
    x1_max = max([p.x1 for p in positions])
    cbar_ax = fig.add_axes([x1_max + 0.015, y0, 0.01, height])

    fig1=ax[0].imshow(v.T, aspect='auto', extent=[0, x.max(), z.max(), 0],
                    cmap='RdBu_r', vmin=1000, vmax=4700)
    ax[0].set_title("Inversion Result")
    ax[0].set_ylabel('Depth (km)')
    ax[0].set_yticks(np.arange(0, z.max(), 0.5))
    ax[0].set_xlabel('Distance (km)')

    fig2=ax[1].imshow(v_true.cpu().T, aspect='auto', extent=[0, x.max(), z.max(), 0],
                    cmap='RdBu_r', vmin=1000, vmax=4700)
    ax[1].set_title("True Result")
    ax[1].set_yticks([])
    ax[1].set_xlabel('Distance (km)')

    cbar = fig.colorbar(fig2, cax=cbar_ax)
    cbar.set_ticks([1500, 2000, 2500, 3000, 3500, 4000, 4500])
    cbar.set_ticklabels(["1.5", '2.0', '2.5', '3.0', '3.5', '4.0', '4.5'])
    cbar_ax.set_ylabel(r'km/s', rotation=90, labelpad=5)
    if save:
        plt.savefig(fr'{file}_picture.jpg', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def network_velocity_profiles(input, model, file, v_init, v_true, positions, depth_sampling_rate=20):
    checkpoint = torch.load(file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    output = model(input).squeeze()
    v = torch.nn.functional.softplus(output + v_init - v_true.min()) + v_true.min()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, position in enumerate(positions):
        true_profile = v_true.T[:, position].numpy()
        inverted_profile = v.detach().T[:, position].numpy()
        coord_label = f'X={position}'
        axes[i].plot(true_profile, np.arange(len(true_profile)) * depth_sampling_rate,
                     'r-', lw=2, label='True Model')

        axes[i].plot(inverted_profile, np.arange(len(true_profile)) * depth_sampling_rate,
                     'b-', lw=2, label='Inverted Model')

        axes[i].invert_yaxis()
        axes[i].set_ylabel('Depth (m)')
        axes[i].set_title(f'Velocity Profile at {coord_label}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)
    for ax in axes:
        ax.set_xlabel('Velocity (m/s)')
    plt.tight_layout()
    plt.show()
    plt.close()

