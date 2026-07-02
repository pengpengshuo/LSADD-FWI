import time
from deepwave import scalar
import torch
import numpy as np
import random

def set_seed(seed=42): #Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def tikhonov_regularization(v: torch.Tensor, order: int = 1) -> torch.Tensor: #Tikhonov regularization

    if order == 1:
        dv_dz = v[1:, :] - v[:-1, :]
        dv_dx = v[:, 1:] - v[:, :-1]
        reg_loss = torch.mean(dv_dz ** 2) + torch.mean(dv_dx ** 2)

    elif order == 2:
        laplacian = (
                -4 * v[1:-1, 1:-1] +
                v[1:-1, :-2] + v[1:-1, 2:] +
                v[:-2, 1:-1] + v[2:, 1:-1]
        )
        reg_loss = torch.mean(laplacian ** 2)

    else:
        raise ValueError("order must be 1 or 2 for Tikhonov regularization")

    return reg_loss

loss_fn=torch.nn.MSELoss()
def convention_train_one_step(model,optimizer,scheduler,observed_data,preliminary,
                              lambda_tikhonov, reg_order,loss_history,epoch,file,save_num=300):
    '''
    lambda_tikhonov:Regularized weights
    reg_order:The order of regularization
    save_num:The number of training sessions saved in the file
    '''
    start_time=time.time()
    optimizer.zero_grad()
    output=model() #The output of the neural network, namely the velocity model
    out=scalar(output,preliminary.dx,preliminary.dt,
               source_amplitudes=preliminary.source_amplitudes,
               source_locations=preliminary.source_locations,
               receiver_locations=preliminary.receiver_locations,
               accuracy=8,
               pml_freq=preliminary.freq
               ) #Generate simulated data
    loss=((0.5*preliminary.n_shots*preliminary.n_receivers_per_shot*preliminary.nt*preliminary.dt)*loss_fn(out[-1],observed_data)
          +lambda_tikhonov*tikhonov_regularization(output,reg_order)) #Loss function with Tikhonov regularization
    loss_history.append(loss.item()) #Store the current training loss
    loss.backward() #Obtain gradient
    optimizer.step() #Update the corresponding parameters
    scheduler.step() #Adjust the learning rate according to the learning rate adjustment strategy
    lr = optimizer.param_groups[0]['lr']
    end_time=time.time()
    execution_time=end_time-start_time
    if (epoch+1)%30==0:
        print(f'this is {epoch + 1}th iteration,time:{execution_time},loss:{loss.item()},lr:{lr}')
    if (epoch + 1) % save_num == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_history':loss_history
        }
        torch.save(checkpoint, f'{file}/train_epoch_{epoch + 1}th.pth')
        print(f'Checkpoint saved at epoch {epoch + 1}')


def batch_train_one_step(model, optimizer, scheduler, input, v_init, v_true, batch_idx, total_batch,
                         observed_data, preliminary,
                         loss_history, epoch, file,start_time, transform=False, save_num=200):
    '''
    input:Input of reparameterized network
    batch_idx:Index of batch size
    transform:Is there a minimum velocity limit
    save_num:The number of training sessions saved in the file
    '''

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    output = model(input).squeeze()
    if transform:
        v = torch.nn.functional.softplus(output + v_init - v_true.min()) + v_true.min() #Set the minimum velocity to ensure the DeepWave works properly
    else:
        v = output + v_init
    #Divide the observation data into multiple batches
    batch_observed_data = observed_data[batch_idx::total_batch]
    source_location = preliminary.source_locations[batch_idx::total_batch]
    source_amplitude=preliminary.source_amplitudes[batch_idx::total_batch]
    receiver_location=preliminary.receiver_locations[batch_idx::total_batch]
    out = scalar(v, preliminary.dx, preliminary.dt,
                 source_amplitudes=source_amplitude,
                 source_locations=source_location,
                 receiver_locations=receiver_location,
                 accuracy=8,
                 pml_freq=preliminary.freq
                 ) #Generate simulated data
    loss = (0.5 * (
                preliminary.n_shots // total_batch) * preliminary.n_receivers_per_shot * preliminary.nt * preliminary.dt) * loss_fn(
        out[-1], batch_observed_data) #Define loss function
    if batch_idx == total_batch-1:
        loss_history.append(loss.item()) #Store the current training loss
    loss.backward() #Obtain gradient
    optimizer.step() #Update the corresponding parameters
    scheduler.step() #Adjust the learning rate according to the learning rate adjustment strategy
    grad_norms = torch.tensor([p.grad.norm() for p in model.parameters() if p.grad is not None])
    total_norm = torch.norm(grad_norms)
    max_norm = max(grad_norms)
    lr = optimizer.param_groups[0]['lr']
    if (epoch+1)%10==0 and batch_idx == total_batch - 1:
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f'epoch:{epoch + 1} | batch:{batch_idx + 1} | time:{execution_time:.3f} | loss:{loss.item():.2f} | lr:{lr:.6f} '
            f'| memory_reseved:{torch.cuda.memory_reserved() / (1024 ** 3):.3f} | max_memory_allocated:{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} | memory_allocated:{torch.cuda.memory_allocated() / (1024 ** 3):.3f} '
            f'| grad_tatal:{total_norm:.3f}, | grad_norm:{max_norm:.3f}')
    if (epoch + 1) % save_num == 0:
        if batch_idx == total_batch-1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss_history': loss_history
            }
            torch.save(checkpoint, f'{file}/train_epoch_{epoch + 1}th.pth')
            print(f'Checkpoint saved at epoch {epoch + 1}')
