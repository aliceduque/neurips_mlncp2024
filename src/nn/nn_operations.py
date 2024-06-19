from ..base.dicts import create_net_dict
from .nn_classes import *
import torch.nn.init as init
import torch.nn as nn


def create_net(activation, noise_on_activation, noise_vec):

    if activation in create_net_dict:
        net_class, learning_rate = create_net_dict[activation]
        net = net_class(noise_on_activation = noise_on_activation, add_unc_var=noise_vec[0], add_corr_var=noise_vec[1],
                        mul_unc_var=noise_vec[2], mul_corr_var=noise_vec[3])
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
    model.add_unc_var = noise_vec[0]
    model.add_corr_var = noise_vec[1]
    model.mul_unc_var = noise_vec[2]
    model.mul_corr_var = noise_vec[3]
    model.noise.update_parameters(model.add_unc_var, model.add_corr_var, model.mul_unc_var, model.mul_corr_var)
    return model
