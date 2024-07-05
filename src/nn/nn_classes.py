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

# class Net_Sigm(nn.Module):
#     def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
#                  add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
#         super(Net_Sigm, self).__init__()
#         self.add_unc_mean = add_unc_mean
#         self.add_unc_var = add_unc_var
#         self.mul_unc_mean = mul_unc_mean
#         self.mul_unc_var = mul_unc_var
#         self.add_corr_mean = add_corr_mean
#         self.add_corr_var = add_corr_var
#         self.mul_corr_mean = mul_corr_mean
#         self.mul_corr_var = mul_corr_var        
#         self.h1 = nn.Linear(INPUT_SIZE, HIDDEN_NEURONS)
#         self.h2 = nn.Linear(HIDDEN_NEURONS,HIDDEN_NEURONS)
#         self.out = nn.Linear(HIDDEN_NEURONS, OUTPUT_SIZE)
#         self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
#                                          mul_unc_mean,add_corr_mean,mul_corr_mean)
#         self.noise_on_activation = noise_on_activation

#     def forward(self, x):
#         x = x.reshape(-1, INPUT_SIZE)
#         if self.noise_on_activation == 'before':
#             x = torch.sigmoid(self.noise(self.h1(x)))
#             x = torch.sigmoid(self.noise(self.h2(x)))
#         elif self.noise_on_activation == 'after':
#             x = self.noise(torch.sigmoid(self.h1(x)))
#             x = self.noise(torch.sigmoid(self.h2(x)))        
#         x = self.out(x)
#         return x


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

# class AddMulGaussianNoise(nn.Module):
#     def __init__(self, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
#                 add_unc_mean=0.0, mul_unc_mean=0.0, add_corr_mean=0.0, mul_corr_mean=0.0):
#         super(AddMulGaussianNoise, self).__init__()
#         self.add_unc_mean = add_unc_mean
#         self.add_unc_var = add_unc_var
#         self.mul_unc_mean = mul_unc_mean
#         self.mul_unc_var = mul_unc_var
#         self.add_corr_mean = add_corr_mean
#         self.add_corr_var = add_corr_var
#         self.mul_corr_mean = mul_corr_mean
#         self.mul_corr_var = mul_corr_var

#     def corr_noise(self, x):
#         dev = x.device
#         M, N = x.size()
#         noise = torch.randn((M, 1),device=dev)
#         noise = noise.expand(M, N)
#         return noise

#     def forward(self, x):
#         dev = x.device
#         # Generate additive noise
#         add_unc = self.add_unc_mean + torch.randn_like(x) * math.sqrt(self.add_unc_var)
#         add_corr = self.add_corr_mean + math.sqrt(self.add_corr_var) * self.corr_noise(x)
#         # Generate multiplicative noise
#         mul_unc = self.mul_unc_mean + torch.randn_like(x) * (math.sqrt(self.mul_unc_var))
#         mul_corr = self.mul_corr_mean + math.sqrt(self.mul_corr_var) * self.corr_noise(x)
#         # print('add unc var: ',self.add_unc_var)
#         # print('add corr var: ',self.add_corr_var)
#         # print('mul corr var: ',self.mul_corr_var)
#         # print('mul corr var: ',self.mul_corr_var)
#         # Apply both additive and multiplicative noise
#         noisy_output = x * (1 + mul_unc) * (1 + mul_corr) + add_unc + add_corr
#         # print(add_unc + add_corr + mul_unc + mul_corr)
#         return noisy_output
    
#     def update_parameters(self, add_unc_var, add_corr_var, mul_unc_var, mul_corr_var):
#         self.add_unc_var = add_unc_var
#         self.add_corr_var = add_corr_var
#         self.mul_unc_var = mul_unc_var        
#         self.mul_corr_var = mul_corr_var

