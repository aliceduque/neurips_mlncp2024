from ..base.dicts import create_net_dict
from ..utils.utils import discretise_tensor
from .nn_classes import *
import torch.nn.init as init
import torch.nn as nn


def create_net(activation, noise_on_activation, noise_vec, noise_no_grad):

    if activation in create_net_dict:
        net_class, learning_rate = create_net_dict[activation]
        net = net_class(noise_on_activation = noise_on_activation, var=noise_vec, noise_no_grad=noise_no_grad,mean=[0,0,0,0])
        learning_rate = learning_rate
    else:
        raise ValueError(f"I don't know that parameter {activation}! Please include it in dicts.py activation map")
    
    return net


def initialize_weights(model, init_type='xavier'):
    def init_func(m):
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                init.xavier_uniform_(m.weight)
            elif init_type == 'he':
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'normal':
                init.normal_(m.weight, mean=0.0, std=0.05)
            elif init_type == 'uniform':
                init.uniform_(m.weight, a=-0.5, b=0.5)
            else:
                raise ValueError(f"Unsupported initialization type: {init_type}")
            
            if m.bias is not None:
                init.zeros_(m.bias)

    model.apply(init_func)


def assign_to_model(model, noise_vec):
    model.var = noise_vec
    # model.noise.update_parameters(var = noise_vec)
    return model

def reg_zero_sum(model):
    total_sum = 0.0
    
    sum_h2 = torch.sum(torch.abs(torch.sum(model.h2.weight,dim=1)))
    sum_out = torch.sum(torch.abs(torch.sum(model.out.weight,dim=1)))
    total_sum = sum_h2 + sum_out

    # for name, param in model.named_parameters():
    #     # Check if the parameter is a weight matrix and not a bias vector
    #     if 'weight' in name:
    #         if 'h1' in name:
    #             continue            
    #         row_sums = torch.sum(param, dim=1)  # Sum elements in each row
    #         abs_row_sums = torch.abs(row_sums)  # Take the absolute value of each row sum
    #         total_sum += torch.sum(abs_row_sums).item()  # Sum these absolute values and add to total_sum
    
    return total_sum

def reg_zero_std(model):
    total_sum = 0.0
    # sum_h1 = torch.sum(torch.std(model.h1.weight,dim=1))
    sum_h2 = torch.sum(torch.std(model.h2.weight,dim=1))
    sum_out = torch.sum(torch.std(model.out.weight,dim=1))
    total_sum = sum_h2 + sum_out
    # total_sum = sum_h2

    # for name, param in model.named_parameters():
    #     # Check if the parameter is a weight matrix and not a bias vector
    #     if 'weight' in name:
    #         if 'h1' in name:
    #             continue            
    #         row_sums = torch.sum(param, dim=1)  # Sum elements in each row
    #         abs_row_sums = torch.abs(row_sums)  # Take the absolute value of each row sum
    #         total_sum += torch.sum(abs_row_sums).item()  # Sum these absolute values and add to total_sum
    
    return total_sum

def regularisation(model, type):
    total_sum = 0.0

    if type == 'custom_sum':
        sum_h2 = torch.sum(torch.abs(torch.sum(model.h2.weight,dim=1)))
        sum_out = torch.sum(torch.abs(torch.sum(model.out.weight,dim=1)))
        reg_factor = sum_h2 + sum_out
        
    elif type == 'custom_std':
        std_h2 = torch.sum(torch.std(model.h2.weight,dim=1))
        std_out = torch.sum(torch.std(model.out.weight,dim=1))
        sum_out = torch.sum(torch.abs(torch.sum(model.out.weight,dim=1)))
        sum_h2 = torch.sum(torch.abs(torch.sum(model.h2.weight,dim=1)))
        reg_factor = 5*std_out + sum_out + 0.1*std_h2 + 0.1*sum_h2

    elif type == 'l1' or type == 'L1':
        reg_factor = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:
                    reg_factor += param.abs().sum()
        
    elif type == 'l2' or type == 'L2':
        reg_factor = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:
                    reg_factor += param.pow(2).sum()

    elif type == 'reg_relu':
        sum_h1 = torch.sum((1/(model.h1.weight**2 + 1e-15)))
        reg_factor = sum_h1
        reg_factor = std_h2 + std_out
        
    elif type == 'custom_bias_out':
        sum_h2 = torch.sum(torch.abs(torch.sum(model.h2.weight,dim=1)))
        std_h2 = torch.sum(torch.std(model.h2.weight,dim=1))
        sum_out = torch.sum(torch.abs(torch.sum(model.out.weight,dim=1)))
        std_out = torch.sum(torch.std(model.out.weight,dim=1))
        # bias_out = torch.sum((1/(model.out.bias**4 + 1e-15)))
        reg_factor = sum_h2 + std_h2 + std_out + sum_out # + bias_out
    
    elif type == 'small_activations':
        reg_factor=0.0
        for name, param in model.named_parameters():
            if 'weight' in name and 'out' not in name:
                reg_factor += torch.sum(torch.clamp(param, min=0))
    
    elif type == None:
        reg_factor = 0    
        
    else:
        raise Warning (f'Regularisation {type} is not recognized')
        reg_factor = 0        
    
    return reg_factor

def discretise_weights (model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'h1' in name:
                continue
            param.data = discretise_tensor(param.data)
            
            
                        