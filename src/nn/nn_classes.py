import torch
import math
import torch.nn as nn
import torch.nn.init as init
from .nn_custom_activations import *
from ..base.init import INPUT_SIZE, OUTPUT_SIZE







class Net_Sigm(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_Sigm, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var        
        self.h1 = nn.Linear(INPUT_SIZE, 30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.sigmoid(self.noise(self.h1(x)))
            x = torch.sigmoid(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.sigmoid(self.h1(x)))
            x = self.noise(torch.sigmoid(self.h2(x)))        
        x = self.out(x)
        return x


class Net_ReLU(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_ReLU, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.relu(self.noise(self.h1(x)))
            x = torch.relu(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.relu(self.h1(x)))
            x = self.noise(torch.relu(self.h2(x)))        
        x = self.out(x)
        return x
    
class Net_ReLU_LN(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_ReLU_LN, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.ln1 = nn.LayerNorm(30)  # Layer normalization for h1 output
        self.ln2 = nn.LayerNorm(30)  # Layer normalization for h2 output
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ln1(torch.relu(self.noise(self.h1(x))))
            x = self.ln2(torch.relu(self.noise(self.h2(x))))
        elif self.noise_on_activation == 'after':
            x = self.ln1(self.noise(torch.relu(self.h1(x))))
            x = self.ln2(self.noise(torch.relu(self.h2(x))))        
        x = self.out(x)

        # x = x.reshape(-1, INPUT_SIZE)
        # x = self.ln1(torch.relu(self.noise(self.h1(x))))
        # x = self.ln2(torch.relu(self.noise(self.h2(x)))) 
        # x = self.out(x)
        return x
    
class Net_ReLU_LN1(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_ReLU_LN1, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.ln1 = nn.LayerNorm(30)  # Layer normalization for h1 output
        self.ln2 = nn.LayerNorm(30)  # Layer normalization for h2 output
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ln1(torch.relu(self.noise(self.h1(x))))
            x = torch.relu(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.ln1(self.noise(torch.relu(self.h1(x))))
            x = self.noise(torch.relu(self.h2(x))) 

        # x = self.out(x)        
        # x = x.reshape(-1, INPUT_SIZE)
        # x = self.ln1(torch.relu(self.noise(self.h1(x))))
        # x = torch.relu(self.noise(self.h2(x)))    
        # x = self.out(x)
        return x



class Net_Leaky(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_Leaky, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.nn.functional.leaky_relu(self.noise(self.h1(x)))
            x = torch.nn.functional.leaky_relu(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.nn.functional.leaky_relu(self.h1(x)))
            x = self.noise(torch.nn.functional.leaky_relu(self.h2(x)))        

        # x = self.out(x)        
        # x = x.reshape(-1, INPUT_SIZE)
        # x = torch.nn.functional.leaky_relu(self.noise(self.h1(x)), negative_slope=0.1)
        # x = torch.nn.functional.leaky_relu(self.noise(self.h2(x)), negative_slope=0.1)
        # x = self.out(x)
        return x
    
class Net_ReLU_bound(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_ReLU_bound, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.ReLU_bound = ReLU_bound()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ReLU_bound(self.noise(self.h1(x)))
            x = self.ReLU_bound(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.ReLU_bound(self.h1(x)))
            x = self.noise(self.ReLU_bound(self.h2(x)))        
        x = self.out(x)
        return x
    
class Net_ReLU_bound_symm(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_ReLU_bound_symm, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.ReLU_bound_symm = ReLU_bound_symm()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.ReLU_bound_symm(self.noise(self.h1(x)))
            x = self.ReLU_bound_symm(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.ReLU_bound_symm(self.h1(x)))
            x = self.noise(self.ReLU_bound_symm(self.h2(x)))        
        x = self.out(x)
        return x


class Net_Erf(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_Erf, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE,30)
        self.h2 = nn.Linear(30,30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var,mul_unc_var,add_corr_var,mul_corr_var,add_unc_mean,
                                         mul_unc_mean,add_corr_mean,mul_corr_mean)
        self.erf = erf()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.erf(self.noise(self.h1(x)))
            x = self.erf(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.erf(self.h1(x)))
            x = self.noise(self.erf(self.h2(x)))        
        x = self.out(x)
        return x


class Net_GELU(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_GELU, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE, 30)
        self.h2 = nn.Linear(30, 30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var, mul_unc_var, add_corr_var, mul_corr_var, 
                                         add_unc_mean, mul_unc_mean, add_corr_mean, mul_corr_mean)
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.nn.functional.gelu(self.noise(self.h1(x)))
            x = torch.nn.functional.gelu(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.nn.functional.gelu(self.h1(x)))
            x = self.noise(torch.nn.functional.gelu(self.h2(x)))        
        x = self.out(x)
        return x
    

class Net_Tanh(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_Tanh, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE, 30)
        self.h2 = nn.Linear(30, 30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var, mul_unc_var, add_corr_var, mul_corr_var, 
                                         add_unc_mean, mul_unc_mean, add_corr_mean, mul_corr_mean)
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = torch.nn.functional.tanh(self.noise(self.h1(x)))
            x = torch.nn.functional.tanh(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(torch.nn.functional.tanh(self.h1(x)))
            x = self.noise(torch.nn.functional.tanh(self.h2(x)))        
        x = self.out(x)
        return x
    

class Net_Sigm_shift(nn.Module):
    def __init__(self, noise_on_activation, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                 add_unc_mean=0, mul_unc_mean=0, add_corr_mean=0, mul_corr_mean=0):
        super(Net_Sigm_shift, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var
        self.h1 = nn.Linear(INPUT_SIZE, 30)
        self.h2 = nn.Linear(30, 30)
        self.out = nn.Linear(30, OUTPUT_SIZE)
        self.noise = AddMulGaussianNoise(add_unc_var, mul_unc_var, add_corr_var, mul_corr_var, 
                                         add_unc_mean, mul_unc_mean, add_corr_mean, mul_corr_mean)
        self.Sigm_shift = Sigm_shift()
        self.noise_on_activation = noise_on_activation

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        if self.noise_on_activation == 'before':
            x = self.Sigm_shift(self.noise(self.h1(x)))
            x = self.Sigm_shift(self.noise(self.h2(x)))
        elif self.noise_on_activation == 'after':
            x = self.noise(self.Sigm_shift(self.h1(x)))
            x = self.noise(self.Sigm_shift(self.h2(x)))        
        x = self.out(x)
        return x    



class AddMulGaussianNoise(nn.Module):
    def __init__(self, add_unc_var=0.0, mul_unc_var=0.0, add_corr_var=0.0, mul_corr_var=0.0,
                add_unc_mean=0.0, mul_unc_mean=0.0, add_corr_mean=0.0, mul_corr_mean=0.0):
        super(AddMulGaussianNoise, self).__init__()
        self.add_unc_mean = add_unc_mean
        self.add_unc_var = add_unc_var
        self.mul_unc_mean = mul_unc_mean
        self.mul_unc_var = mul_unc_var
        self.add_corr_mean = add_corr_mean
        self.add_corr_var = add_corr_var
        self.mul_corr_mean = mul_corr_mean
        self.mul_corr_var = mul_corr_var

    def corr_noise(self, x):
        dev = x.device
        M, N = x.size()
        noise = torch.randn((M, 1),device=dev)
        noise = noise.expand(M, N)
        return noise

    def forward(self, x):
        dev = x.device
        # Generate additive noise
        add_unc = self.add_unc_mean + torch.randn_like(x) * math.sqrt(self.add_unc_var)
        add_corr = self.add_corr_mean + math.sqrt(self.add_corr_var) * self.corr_noise(x)
        # Generate multiplicative noise
        mul_unc = self.mul_unc_mean + torch.randn_like(x) * (math.sqrt(self.mul_unc_var))
        mul_corr = self.mul_corr_mean + math.sqrt(self.mul_corr_var) * self.corr_noise(x)
        # print('add unc var: ',self.add_unc_var)
        # print('add corr var: ',self.add_corr_var)
        # print('mul corr var: ',self.mul_corr_var)
        # print('mul corr var: ',self.mul_corr_var)
        # Apply both additive and multiplicative noise
        noisy_output = x * (1 + mul_unc) * (1 + mul_corr) + add_unc + add_corr
        # print(add_unc + add_corr + mul_unc + mul_corr)
        return noisy_output
    
    def update_parameters(self, add_unc_var, add_corr_var, mul_unc_var, mul_corr_var):
        self.add_unc_var = add_unc_var
        self.add_corr_var = add_corr_var
        self.mul_unc_var = mul_unc_var        
        self.mul_corr_var = mul_corr_var

