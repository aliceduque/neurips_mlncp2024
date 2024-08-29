import torch
import torch.nn as nn
import torch.autograd as autograd


class erf(nn.Module):
    def forward(self, x):
        return torch.erf(x)

class ReLU_bound(nn.Module):
    def forward(self, x):
        y = torch.zeros_like(x)
        a = 8
        y[(x <= 0)] = 0
        y[(x > 0) & (x < a)] = x[(x > 0) & (x < a)]/a
        y[(x >= a)] = 1
        return y

class ReLU_bound_symm(nn.Module):
    def forward(self, x):
        y = torch.zeros_like(x)
        a = 8
        y[(x <= -(a/2))] = 0
        y[(x > -(a/2)) & (x < (a/2))] = x[(x > (-a/2)) & (x < (a/2))]/a + 0.5
        y[(x >= (a/2))] = 1
        return y
    
class Sigm_shift(nn.Module):
    def forward(self, x):
        y = torch.sigmoid(x-4)
        return y
    
class PhotonicSigmoidFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save the input tensor for backward computation
        ctx.save_for_backward(x)
        if torch.isnan(x).any():   
            print(f"NaN detected in forward INPUT")
        
        # Forward pass: element-wise condition
        y = torch.where(
            x > 20, 
            torch.tensor(1.005, device=x.device, dtype=x.dtype), 
            1.005 + (0.06 - 1.005) / (1 + torch.exp((x - 0.145) / 0.033))
        )
        
        if torch.isnan(y).any():
            nan_indices = torch.nonzero(torch.isnan(y))            
            print(f"NaN detected in forward, for x = {x[nan_indices[:, 0]]}")
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        
        def derivative(x):
            exp_part = torch.exp((x - 0.145) / 0.033)
            deriv = (-exp_part / ((1 + exp_part) ** 2)) * ((0.06 - 1.005) / 0.033)
            return deriv
        
        x, = ctx.saved_tensors
        if torch.isnan(grad_output).any():
            nan_indices = torch.nonzero(torch.isnan(grad_output))     
            print(f"NaN detected in backwards, grad_out = {grad_output[nan_indices[:, 0]] }")
        # Compute the gradient of the sigmoid part for values within range
        
        # if torch.isnan(sigmoid_derivative).any():          
        #     print(f"NaN detected in backwards (sigm_derivative)")
        # Apply the condition: gradient should be zero for x > 10
        grad_input = torch.where(x > 2, torch.tensor(1e-25, device=x.device, dtype=x.dtype), derivative(x))

        if torch.isnan(grad_input).any():
            nan_indices = torch.nonzero(torch.isnan(grad_input))
            # print(nan_indices)            
            print(f"NaN detected in backwards, for grad_out = {x[nan_indices[:, 0]] }")
        
        # Element-wise multiplication with grad_output (chain rule)
        grad_input = grad_input * grad_output
        # print('grad: ', grad_input)

            # Get the indices of NaN values

        return grad_input
    

class Photonic_sigmoid(nn.Module):
    def forward(self, x):
        return PhotonicSigmoidFunction.apply(x)