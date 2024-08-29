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

    
    return total_sum

def reg_zero_std(model):
    total_sum = 0.0
    # sum_h1 = torch.sum(torch.std(model.h1.weight,dim=1))
    sum_h2 = torch.sum(torch.std(model.h2.weight,dim=1))
    sum_out = torch.sum(torch.std(model.out.weight,dim=1))
    total_sum = sum_h2 + sum_out
    # total_sum = sum_h2

    
    return total_sum

class ActivationHook:
    def __init__(self, layer, hook_type='output'):
        self.activations = None
        self.hook_type = hook_type

        if hook_type == 'input':
            self.hook = layer.register_forward_hook(self.hook_input_fn)
        elif hook_type == 'output':
            self.hook = layer.register_forward_hook(self.hook_output_fn)
        else:
            raise ValueError("hook_type must be either 'input' or 'output'")

    def hook_input_fn(self, module, input, output):
        self.activations = input[0]  # input is a tuple, take the first element

    def hook_output_fn(self, module, input, output):
        self.activations = output

    def close(self):
        self.hook.remove()

# def add_hook(layer):
#     hook = ActivationHook(layer)
#     hooks.append(hook)
#     print(hook.activations)

def compute_penalty(hooks, type):
    total_penalty = 0.0
    for hook in hooks:
        if hook.activations is not None:
            total_penalty += regularization_term(hook.activations, type)
            
    return total_penalty

def regularization_term(activations, type):
    def sigmoid_derivative(x):
        sigmoid = 1 / (1 + torch.exp(-x))
        return sigmoid * (1 - sigmoid)
    
    if type == 'addunc_sigm':
        penalty = sigmoid_derivative(activations)
        
    elif type == 'addunc_phot_sigm':
        photon = Photonic_derivative_reg()
        penalty = photon.forward(activations)
        
    if torch.isnan(penalty).any():
        print(f"NaN detected in Penalty") 
    
    return penalty.abs().sum()

def regularisation(model, type, hook=None, reg_config=[0,0,0]):
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
        reg_factor = 1.2*std_h2 + 0.1*sum_h2 + 27*std_out + 0.3*sum_out

    elif type == 'towards_saturation':
        hooks = []
        hooks.append(hook)        
        reg_factor = compute_penalty(hooks)
        
    elif type =='addunc_sigm' or type =='addunc_phot_sigm':
        l2_factor_out = model.out.weight.pow(2).sum()
        l2_factor_h2 = model.h2.weight.pow(2).sum()
        sum_h2 = torch.sum(torch.abs(torch.sum(model.h2.weight,dim=1)))
        hooks = []
        hooks.append(hook)        
        reg_factor = reg_config[0]*compute_penalty(hooks, type) + reg_config[1]*l2_factor_h2 + reg_config[2]*l2_factor_out
        
        
    elif type =='h2_saturation_out_l1':
        l1_factor = model.out.weight.abs().sum()
        hooks = []
        hooks.append(hook)        
        reg_factor = compute_penalty(hooks) + l1_factor
        
        
    elif type == 'custom_std_row':
        out = model.out.weight
        row_diff = torch.zeros_like(out)
        for i in range(len(out)):
            if i == 0:
                row_diff[0] = out[0] - out[len(out)-1]
            else:
                row_diff[i] = out[i] - out[i-1]
        # std_out = torch.sum(torch.std(model.out.weight,dim=0))
        # std_h2 = torch.sum(torch.std(model.h2.weight,dim=1))
        row_diff_std = torch.std(row_diff, dim=1)
        reg_factor = row_diff_std.sum()

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
            
            
class Photonic_derivative_function(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        factor = 1
        def derivative(x):
            exp_part = torch.exp((x - 0.145) / 0.033)
            deriv = (-exp_part / ((1 + exp_part) ** 2)) * ((0.06 - 1.005) / 0.033)
            return deriv

        derivative = torch.where(x > 2, torch.tensor(1e-28, device=x.device, dtype=x.dtype), derivative(x))
        return derivative
    
    @staticmethod
    def backward(ctx, grad_output):
        def second_derivative(x):
            first_term = ((18.3655 * torch.exp(6.06061*(x - 0.145)))/(torch.exp(3.0303*(x - 0.145)) + 1).pow(3))
            second_term = (9.18274 * torch.exp(3.0303*(x - 0.145)))/(torch.exp(3.0303*(x - 0.145)) + 1).pow(2)
            result = -0.945 * (first_term - second_term)
            return result
                        
        x, = ctx.saved_tensors
        grad_input = torch.where(x > 14, torch.tensor(1e-19, device=x.device, dtype=x.dtype), second_derivative(x))
        grad_input = grad_input * grad_output

        return grad_input


class Photonic_derivative_reg(nn.Module):
    def forward(self, x):
        return Photonic_derivative_function.apply(x)
