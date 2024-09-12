# Train and test network

import torch
import numpy as np
import torch.nn as nn
from ..data.dataset_ops import get_test_loader, get_train_loader
from ..base.init import INPUT_SIZE,OUTPUT_SIZE
from ..graphic.plot_ops import plot_loss_curve, plot_gradients
from ..graphic.simplex_ops import create_simplex_shaded, plot_simplex
from ..utils.utils import get_actual_output, expand_expected_output
from ..base.dicts import gradients
from ..nn.nn_operations import regularisation, ActivationHook
from torchviz import make_dot
import os



def run_network(net, database, root, num_epochs, lr, train=True, test=True, reduced=False, plot=False):
    dev = next(net.parameters()).device
    if train:
      train_loader, validation_loader = get_train_loader(database=database, root=root, reduced=reduced, device=dev)
      cross_entropy_loss = nn.CrossEntropyLoss()
      cross_entropy_loss.to(dev)
      loss_function = create_loss_function(cross_entropy_loss)
      optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
      print('learning rate: ',lr)
      ax = train_network(net, train_loader, num_epochs, loss_function, optimizer, validation_loader, plot=plot)
    if test:
      test_loader = get_test_loader(database=database, root=root, reduced=reduced, device=dev)
      test_network(net, test_loader)
    return ax


def train_network(model, train_loader, num_epochs, loss_function,optimizer, 
                  validation_loader, hook=None, reg_type=None, lambda_reg=0, reg_config = [0,0,0],
                  plot_curve=False, plot_gradient=False):
    if reg_type == 'addunc_sigm' or reg_type == 'addunc_phot_sigm':
        hook = ActivationHook(model.h2, hook_type='output')
    else:
        hook = None
    gradients = {}
    dev = next(model.parameters()).device
    training_losses = []
    validation_accuracies = []
    train_accuracies = []
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        hooks = []
        hooks.append(hook) 
        model.train()
        epoch_loss = 0
        reg_epoch_loss = 0
        epoch_gradients = {}
        for i, (images, expected_outputs) in enumerate(train_loader):

            images, expected_outputs = images.to(dev, non_blocking=True), expected_outputs.to(dev, non_blocking=True)
            outputs = model(images)
            loss = loss_function(outputs, expected_outputs)
            reg_factor = regularisation(model, reg_type, hook, reg_config)
            # torch.autograd.set_detect_anomaly(True)
                             
            total_loss = loss + lambda_reg * reg_factor

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            
            
            for name, param in model.named_parameters():
                # print(f'grad max {name} = {max(param.grad.flatten())}')
                if torch.isnan(param.grad).any():
                    print(f"GRADIENT NaN detected in {name}")  
                if 'weight' in name and param.grad is not None:
                    if name not in epoch_gradients:
                        epoch_gradients[name] = param.grad.clone().detach()
                    else:
                        epoch_gradients[name] += param.grad.clone().detach()

            
            # print(f'{i}, epoch {epoch}, max weight: {max(model.h1.weight.flatten())}')
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{i}, Gradient before clipping ({name}): {param.grad}")
            # for param in model.parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm(2).item()
            # print(f'iteration {i},  {grad_norm}')
            # print('loss : ', loss)

            # print(f'after clip: {model.out.weight.grad}')
            optimizer.step()
            reg_epoch_loss += lambda_reg * reg_factor
            epoch_loss += total_loss
            
        for name, grad in epoch_gradients.items():
            if name not in gradients:
                gradients[name] = []
            avg_grad = grad / num_batches
            gradients[name].append(avg_grad)

        avg_reg_loss = reg_epoch_loss / num_batches
        avg_training_loss = epoch_loss / num_batches
        training_losses.append(avg_training_loss)
        _, validation_accuracy, _ = test_network(model, validation_loader)
        _, train_accuracy, _ = test_network(model, train_loader)
        validation_accuracies.append(validation_accuracy)
        train_accuracies.append(train_accuracy) 
        print('Epoch [{}/{}], Training Loss: {:.4f} (of which reg = {:.4f}), Validation Accuracy: {:.2f}%'.format(
            epoch + 1, num_epochs, avg_training_loss, avg_reg_loss, validation_accuracy))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_norm = torch.norm(param).item()
                    print(f'Epoch [{epoch+1}/{num_epochs}], Layer: {name}, L2 Norm: {l2_norm:.4f}')


    if plot_curve:
        fig = plot_loss_curve(num_epochs,training_losses,train_accuracies,validation_accuracies)
    else:
       fig = None

    if plot_gradient:
        fig2 = plot_gradients(gradients, num_epochs)
    else:
        fig2= None

    return fig, fig2

def save_grad(name):
    def hook(grad):
        if name not in gradients:
            gradients[name] = grad.clone().detach()
        else:
            gradients[name] += grad.clone().detach()
    return hook




def test_network(model, data_loader, simplex=False, epoch_info=""):
    dev = next(model.parameters()).device
    model = model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        if simplex:
          ax, fig = create_simplex_shaded()
        for batch in data_loader:
            images, expected_outputs = batch
            images, expected_outputs = images.to(dev, non_blocking=True), expected_outputs.to(dev, non_blocking=True)
            outputs = model(images)
            expected_classes = expected_outputs.type(torch.int)
            predicted_index = torch.argmax(outputs, dim=1).type(torch.int)
            predicted_outputs = get_actual_output(predicted_index)
            if simplex:
              ax = plot_simplex(outputs,expected_classes,ax)
            else:
              ax = None
            correct += (predicted_outputs == expected_outputs).sum()
            total += expected_outputs.size(0)

        results_str = f"Test data results: {float(correct)/total}"
        if epoch_info:
            results_str += f", {epoch_info}"

        print(float(correct)/total)
    return batch, 100*float(correct)/total, ax


def create_loss_function(loss_function, output_size=OUTPUT_SIZE):
    def calc_loss(outputs, target):
        targets = expand_expected_output(target, output_size)
        return loss_function(outputs, targets)
    return calc_loss


def save_grad(name):
    def hook(grad):
        if name not in gradients:
            gradients[name] = grad.clone().detach()
        else:
            gradients[name] += grad.clone().detach()
    return hook