import torch
import math
import torch.nn as nn
import torch.nn.init as init
from .nn_custom_activations import *
from ..base.init import INPUT_SIZE, OUTPUT_SIZE, HIDDEN_NEURONS





class Net_Sigm(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_Sigm, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE,HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS,HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.sigmoid(self.noise(self.h1(x), self.var, self.mean))
            x = torch.sigmoid(self.noise(self.h2(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.sigmoid(self.h1(x)), self.var, self.mean)
            x = self.noise(torch.sigmoid(self.h2(x)), self.var, self.mean)        
        x = self.out(x)
        return x


class Net_ReLU(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_ReLU, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE,HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS,HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.relu(self.noise(self.h1(x), self.var, self.mean))
            x = torch.relu(self.noise(self.h2(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.relu(self.h1(x)), self.var, self.mean)
            x = self.noise(torch.relu(self.h2(x)), self.var, self.mean)        
        x = self.out(x)
        return x
    
class Net_ReLU_LN(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_ReLU_LN, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE,HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS,HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.ln1 = nn.LayerNorm(HIDDEN_NEURONS)  # Layer normalization for h1 output
        self.ln2 = nn.LayerNorm(HIDDEN_NEURONS)  # Layer normalization for h2 output
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ln1(torch.relu(self.noise(self.h1(x), self.var, self.mean)))
            x = self.ln2(torch.relu(self.noise(self.h2(x), self.var, self.mean)))
        elif self.noise_on_activation == 'after':
            x = self.ln1(self.noise(torch.relu(self.h1(x)), self.var, self.mean))
            x = self.ln2(self.noise(torch.relu(self.h2(x)), self.var, self.mean))        
        x = self.out(x)

        return x
    
class Net_ReLU_LN1(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_ReLU_LN1, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.ln1 = nn.LayerNorm(HIDDEN_NEURONS)  # Layer normalization for h1 output        
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ln1(torch.relu(self.noise(self.h1(x), self.var, self.mean)))
            x = torch.relu(self.noise(self.h2(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.ln1(self.noise(torch.relu(self.h1(x)), self.var, self.mean))
            x = self.noise(torch.relu(self.h2(x)), self.var, self.mean)    
        x = self.out(x)
        return x        
    

class Net_Leaky(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_Leaky, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.leaky= torch.nn.functional.leaky_relu()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.leaky(self.noise(self.h1(x), self.var, self.mean))
            x = self.leaky(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.leaky(self.h1(x)), self.var, self.mean)
            x = self.noise(self.leaky(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x    
            
    
class Net_ReLU_bound(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_ReLU_bound, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.ReLU_bound = ReLU_bound()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ReLU_bound(self.noise(self.h1(x), self.var, self.mean))
            x = self.ReLU_bound(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.ReLU_bound(self.h1(x)), self.var, self.mean)
            x = self.noise(self.ReLU_bound(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x    
        
    
class Net_ReLU_bound_symm(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_ReLU_bound_symm, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.ReLU_bound_symm = ReLU_bound_symm()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ReLU_bound_symm(self.noise(self.h1(x), self.var, self.mean))
            x = self.ReLU_bound_symm(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.ReLU_bound_symm(self.h1(x)), self.var, self.mean)
            x = self.noise(self.ReLU_bound_symm(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x    
        

class Net_Erf(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_Erf, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.erf = erf()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.erf(self.noise(self.h1(x), self.var, self.mean))
            x = self.erf(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.erf(self.h1(x)), self.var, self.mean)
            x = self.noise(self.erf(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x    
    
    
class Net_GELU(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_GELU, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.gelu = torch.nn.functional.gelu
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.gelu(self.noise(self.h1(x), self.var, self.mean))
            x = self.gelu(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.gelu(self.h1(x)), self.var, self.mean)
            x = self.noise(self.gelu(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x    
    

class Net_Tanh(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_Tanh, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.tanh = torch.nn.functional.tanh
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.tanh(self.noise(self.h1(x), self.var, self.mean))
            x = self.tanh(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.tanh(self.h1(x)), self.var, self.mean)
            x = self.noise(self.tanh(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x
    
class Net_Photon_Sigm(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_Photon_Sigm, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.phot_sigm = Photonic_sigmoid()
        self.noise_on_activation = noise_on_activation
        
    def check_for_nan(self, tensor, name=""):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")    
        
    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.phot_sigm(self.noise(self.h1(x), self.var, self.mean))
            x = self.phot_sigm(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.h1(x)
            self.check_for_nan(x, 'After input weights')
            x = self.phot_sigm(x)
            self.check_for_nan(x, 'After 1st activation')
            x = self.noise(x, self.var, self.mean)
            self.check_for_nan(x, 'After 1st noise')
            x = self.h2(x)
            self.check_for_nan(x, 'After h2')
            x = self.phot_sigm(x)
            self.check_for_nan(x, 'After 2nd activation')
            x = self.noise(x, self.var, self.mean)
            self.check_for_nan(x, 'After 2nd noise')
                        
            # x = self.noise(self.phot_sigm(self.h1(x)), self.var, self.mean)
            # x = self.noise(self.phot_sigm(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x    

class Net_Sigm_shift(nn.Module):
    def __init__(self, noise_on_activation, var=[0,0,0,0], mean=[0,0,0,0], noise_no_grad = False):
        super(Net_Sigm_shift, self).__init__()
        self.var = var
        self.mean = mean
        self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
        self.h2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
        if noise_no_grad:
            self.noise = AddMulGaussianNoise_nograd.apply
        else:
            self.noise = AddMulGaussianNoise.apply
        self.sigm_shift = Sigm_shift()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.sigm_shift(self.noise(self.h1(x), self.var, self.mean))
            x = self.sigm_shift(self.noise(self.h1(x), self.var, self.mean))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.sigm_shift(self.h1(x)), self.var, self.mean)
            x = self.noise(self.sigm_shift(self.h2(x)), self.var, self.mean)      
        x = self.out(x)
        return x


class AddMulGaussianNoise_nograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, var=[0,0,0,0], mean = [0,0,0,0]):
        add_unc_var = var[0]
        add_corr_var = var[1]
        mul_unc_var = var[2]
        mul_corr_var = var[3]
        add_unc_mean = mean[0]
        add_corr_mean = mean[1]
        mul_unc_mean = mean[2]
        mul_corr_mean = mean[3]      
        
        ctx.save_for_backward(x)

        def corr_noise(x):
            dev = x.device
            M, N = x.size()
            noise = torch.randn((M, 1),device=dev)
            noise = noise.expand(M, N)
            return noise
            
        add_unc = add_unc_mean + torch.randn_like(x) * math.sqrt(add_unc_var)
        add_corr = add_corr_mean + math.sqrt(add_corr_var) * corr_noise(x)
        mul_unc = mul_unc_mean + torch.randn_like(x) * (math.sqrt(mul_unc_var))
        mul_corr = mul_corr_mean + math.sqrt(mul_corr_var) * corr_noise(x)
        noisy_output = x * (1 + mul_unc.detach()) * (1 + mul_corr.detach()) + add_unc.detach() + add_corr.detach()
        
        return noisy_output


    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input *= input
        return grad_input, None, None
    
    def update_parameters(self, var):
        self.var = var


class AddMulGaussianNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, var=[0, 0, 0, 0], mean=[0, 0, 0, 0]):
        add_unc_var = var[0]
        add_corr_var = var[1]
        mul_unc_var = var[2]
        mul_corr_var = var[3]
        add_unc_mean = mean[0]
        add_corr_mean = mean[1]
        mul_unc_mean = mean[2]
        mul_corr_mean = mean[3]
        
        ctx.save_for_backward(x)
        
        def corr_noise(x):
            dev = x.device
            M, N = x.size()
            noise = torch.randn((M, 1), device=dev)
            noise = noise.expand(M, N)
            return noise
            
        add_unc = add_unc_mean + torch.randn_like(x) * math.sqrt(add_unc_var)
        add_corr = add_corr_mean + math.sqrt(add_corr_var) * corr_noise(x)
        mul_unc = mul_unc_mean + torch.randn_like(x) * math.sqrt(mul_unc_var)
        mul_corr = mul_corr_mean + math.sqrt(mul_corr_var) * corr_noise(x)
        
        noisy_output = x * (1 + mul_unc.detach()) * (1 + mul_corr.detach()) + add_unc.detach() + add_corr.detach()
        
        # Save additional tensors for the backward pass
        ctx.save_for_backward(x, mul_unc, mul_corr)
        
        return noisy_output

    @staticmethod
    def backward(ctx, grad_output):
        x, mul_unc, mul_corr = ctx.saved_tensors
        
        # Compute gradient with respect to x
        grad_x = grad_output * (1 + mul_unc) * (1 + mul_corr)
        
        # Return gradients for x and None for var and mean
        return grad_x, None, None
    def update_parameters(self, var):
        self.var = var
        
        
