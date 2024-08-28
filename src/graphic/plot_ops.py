# Plot functions

import matplotlib.pyplot as plt
import numpy as np
import random
from src.base.dicts import *
from src.utils.utils import noise_label
from src.nn.nn_operations import ActivationHook
from src.nn.nn_operations import assign_to_model



def plot_gradients(gradients, num_epochs):
    fig = plt.figure(figsize=(18,8))
    for idx, (name, grads) in enumerate(gradients.items()):
        flat_grad = [item for sublist1 in grads for sublist2 in sublist1 for item in sublist2]
        bins = np.linspace(min(flat_grad), max(flat_grad), 30) 
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        for epoch in range(num_epochs):
            hist, edges = np.histogram(gradients[name][epoch], bins=bins)
            # print('epoch ', epoch, edges)
            x = 0.5 * (edges[1:] + edges[:-1])
            y = np.ones_like(x) * epoch
            z = hist
            width = (max(flat_grad)-min(flat_grad))/25
            ax.bar(x, z, zs=epoch, zdir='y', alpha=0.8, width=width)
        ax.set_xlabel('Gradient Value')
        ax.set_ylabel('Epoch')
        ax.set_zlabel('Frequency')
        ax.set_title(rf"Average gradients: {name}")
    plt.tight_layout()
    return fig


def plot_loss_curve (num_epochs, training_losses, validation_accuracies):
    for i,loss in enumerate(training_losses):
        training_losses[i] = loss.detach().cpu().numpy()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(range(1, num_epochs + 1), training_losses, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:orange', fontsize=14)
    ax2.plot(range(1, num_epochs + 1), validation_accuracies, label='Validation Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.grid(True)
    return fig


def scatter_traces(noise_variance, values, ylabel='', title='', xlabel='Noise variance',
                ret = False):

    colors = ['blue', 'green', 'red', 'orange']
    labels = ['Additive Uncorrelated', 'Additive Correlated', 'Multiplicative Uncorrelated', 'Multiplicative Correlated']
    fig,ax = plt.subplots()
    values = values.detach().cpu().numpy()
    for k in range(len(values)):
        plt.scatter(noise_variance, values[k], color=colors[k], label=labels[k])

    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if ret:
        return fig
    else:
        plt.show()

def plot_traces_fill(noise_variance, values, ylabel='', title='', xlabel='Noise variance',
                ret = False):
  
    fig,ax = plt.subplots()
    colors = ['blue', 'green', 'red', 'orange']
    labels = ['Additive Uncorrelated', 'Additive Correlated', 'Multiplicative Uncorrelated', 'Multiplicative Correlated']
    values = values.detach().cpu().numpy()
    ax.xscale('log')
    ax.xlabel(xlabel)
    ax.ylabel(ylabel)
    ax.title(title)
    for k in range(len(values)):
        ax.plot(noise_variance, np.mean(values[k], axis=1), color=colors[k], label=labels[k], linewidth = 3)
        ax.fill_between(noise_variance, np.mean(values[k], axis=1) - np.std(values[k], axis=1),
                        np.mean(values[k], axis=1) + np.std(values[k], axis=1), color=colors[k], alpha=0.3)
    ax.legend()
    plt.grid(True)
    if ret:
        return fig
    else:
        plt.show()


def weights_histogram(net, title):
    net_copy = net
    weight_h1 = net_copy.h1.weight.detach().cpu().numpy().flatten()
    weight_h2 = net_copy.h2.weight.detach().cpu().numpy().flatten()
    weight_out = net_copy.out.weight.detach().cpu().numpy().flatten()

    ax = plt.figure(figsize=(8,10))

    plt.suptitle(title)
    plt.subplot(3, 1, 1)
    plt.hist(weight_h1, bins=20, color='blue', alpha=0.7)
    plt.title('Layer 1 weight Histogram')

    plt.subplot(3, 1, 2)
    plt.hist(weight_h2, bins=20, color='blue', alpha=0.7)
    plt.title('Layer 2 weight Histogram')

    plt.subplot(3,1, 3)
    plt.hist(weight_out, bins=20, color='blue', alpha=0.7)
    plt.title('Output Layer weight Histogram')

    plt.tight_layout()

    return ax

def biases_histogram(net, title):
    net_copy = net
    bias_h1 = net_copy.h1.bias.detach().cpu().numpy().flatten()
    bias_h2 = net_copy.h2.bias.detach().cpu().numpy().flatten()
    bias_out = net_copy.out.bias.detach().cpu().numpy().flatten()

    ax = plt.figure(figsize=(8,10))

    plt.suptitle(title)
    plt.subplot(3, 1, 1)
    plt.hist(bias_h1, bins=20, color='green', alpha=0.7)
    plt.title('Layer 1 bias Histogram')

    plt.subplot(3, 1, 2)
    plt.hist(bias_h2, bins=20, color='green', alpha=0.7)
    plt.title('Layer 2 bias Histogram')

    plt.subplot(3,1, 3)
    plt.hist(bias_out, bins=20, color='green', alpha=0.7)
    plt.title('Output Layer bias Histogram')

    plt.tight_layout()

    return ax

def weights_mean_std(model, title):
    
    def get_weight_tensors(model):
        weights_tensors = []
        bias_tensors = []

        for param in model.parameters():
            if param.requires_grad and len(param.shape) >= 2:
                weights_tensors.append(param.data.clone())
            elif param.requires_grad and len(param.shape) == 1:
                bias_tensors.append(param.data.clone())
        
        return weights_tensors, bias_tensors
    
    def extract_mean_std(weights_tensors, row_or_col, bias_tensors):
        if row_or_col == 'row':
            dim = 1
        elif row_or_col == 'col':
            dim = 0
        
        layer_means = []
        layer_std_devs = []
        bias = []
        
        for tensor, bias_tensor in zip(weights_tensors, bias_tensors):
            means = tensor.mean(dim=dim).numpy()
            variances = tensor.var(dim=dim).numpy()
            std_devs = np.sqrt(variances)
            sorted_indices = means.argsort()
            print(len(sorted_indices))
            sorted_means = means[sorted_indices]
            sorted_std_devs = std_devs[sorted_indices]
            
            layer_means.append(sorted_means)
            layer_std_devs.append(sorted_std_devs)
            if row_or_col == 'row':
                sorted_bias = bias_tensor[sorted_indices]
                bias.append(sorted_bias)  
        return layer_means, layer_std_devs, bias    
        
         
    def plot_hist_rows_columns(layer_means, layer_std_devs, row_or_col, title, bias = None):
        
        num_layers = len(layer_means)
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 16), sharex=False)
        colors = plt.cm.brg(np.linspace(0, 1, num_layers))

        for i in range(num_layers):
            ax = axes[i]
            x_values = range(len(layer_means[i]))
            print(x_values)
            ax.plot(x_values, layer_means[i], color=colors[i], label=f'Layer {i+1} Mean', linewidth=3)
            ax.fill_between(range(len(layer_means[i])),
                            layer_means[i] - 2*layer_std_devs[i],
                            layer_means[i] + 2*layer_std_devs[i],
                            color=colors[i], alpha=0.2)
            if bias is not None:
                ax.bar(x_values, bias[i], color='gray', alpha=0.4, label=f'layer {i+1} Bias')
            ax.set_ylabel('Values')
            ax.legend()
            ax.grid(True)
            ax.set_title(f'Layer {i+1}')

        axes[-1].set_xlabel(f'{row_or_col} #')
        fig.suptitle(f'Mean Values and Standard Deviation of {row_or_col}s for {title}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    
    weights_tensors, bias_tensors = get_weight_tensors(model)
    
    row_means, row_std_devs, bias = extract_mean_std(weights_tensors, 'row', bias_tensors = bias_tensors)
    col_means, col_std_devs, _ = extract_mean_std(weights_tensors, 'col', bias_tensors = bias_tensors)
    
    fig_row = plot_hist_rows_columns(row_means, row_std_devs, 'row', title, bias = bias)
    fig_col = plot_hist_rows_columns(col_means, col_std_devs, 'column', title)
   
    return fig_row, fig_col


def make_plot(activation, attribute, value, train_vec, noise_points, noise_range, test_noises, noise_on_activation):
    train_vec = train_vec.cpu().numpy()
    value = value.cpu().numpy()
    noise_vec = np.logspace(np.log10(noise_range[0]), np.log10(noise_range[1]), noise_points)    
    colors = [color_dict[v] for v in test_noises]
    labels = [label_dict[v] for v in test_noises]
    noise_labels = noise_label(train_vec)
    noise_value = np.amax(train_vec)
    title = (rf'{attribute}: {activation} trained with {noise_labels} = {noise_value:.1f} {noise_on_activation} activation')
    fig, ax = plt.subplots()
    mean = np.mean(value, axis=0)
    std = np.std(value, axis=0)
    for k in range(len(test_noises)):
        ax.plot(noise_vec, mean[:,k], color = colors[k], label = labels[k], linewidth=3)
        ax.fill_between(noise_vec, mean[:,k] - std[:,k], mean[:,k] + std[:,k], color=colors[k], alpha=0.3)

    ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel('Noise variance')
    ax.set_ylabel(attribute)
    ax.legend()
    plt.grid(True)
    return fig




   
def activations_plot(model, test_dataset):

    def create_hooks(model, layers, hook_type='output'):
        hooks = {}
        for name, layer in layers.items():
            hooks[f'{name}_{hook_type}'] = ActivationHook(layer, hook_type)
        return hooks

    def plot_histogram_and_save(data, layer_name, hook_type, color='red'):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, color=color, alpha=0.7)   
        plt.title(f'Histogram of {layer_name} Layer ({hook_type}) Activations')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'{layer_name}_{hook_type}.png')
        plt.close()
        
    assign_to_model(model, [0,0,0,0])

    layers_to_hook = {'h1': model.h1, 'h2': model.h2, 'out': model.out}
    input_hooks = create_hooks(model, layers_to_hook, hook_type='input')
    output_hooks = create_hooks(model, layers_to_hook, hook_type='output')
    all_hooks = {**input_hooks, **output_hooks}

    num = random.randint(0,50)
    for i,batch in enumerate(test_dataset):
        if i == num:
            images, _ = batch
            break

    output = model(images)

    for name, hook in all_hooks.items():
        layer_name, hook_type = name.split('_') 
        plot_histogram_and_save(hook.activations.detach().cpu().numpy().flatten(), layer_name, hook_type)
        if hook_type == 'input':
            if layer_name == 'h1':
                continue
            noise = np.random.normal(0, 1, hook.activations.shape)
            layer = layers_to_hook[layer_name]
            weights = layer.weight.data.detach().numpy()
            print(weights.shape)
            bias = layer.bias.data.detach().numpy()
            print(bias.shape)
            noise_out = np.matmul(noise, weights.T) + bias
            plot_histogram_and_save(noise_out.flatten(), layer_name, 'noise (var = 1.0)', 'brown')

    for hook in all_hooks.values():
        hook.close()