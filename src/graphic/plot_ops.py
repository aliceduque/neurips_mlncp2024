# Plot functions

import matplotlib.pyplot as plt
import numpy as np
from src.base.dicts import *
from src.utils.utils import noise_label


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
        
        # Iterate over all parameters in the model
        for param in model.parameters():
            if param.requires_grad and len(param.shape) >= 2:
                weights_tensors.append(param.data.clone())  # Clone to detach from computation graph
        
        return weights_tensors
    
    weights_tensors = get_weight_tensors(model)
    num_layers = len(weights_tensors)
    
    layer_row_means = []
    layer_row_std_devs = []
    layer_col_means = []
    layer_col_std_devs = []

    for tensor in weights_tensors:
        row_means = tensor.mean(dim=1).numpy()
        row_variances = tensor.var(dim=1).numpy()
        row_std_devs = np.sqrt(row_variances)
        sorted_indices = row_means.argsort()
        sorted_means = row_means[sorted_indices]
        sorted_std_devs = row_std_devs[sorted_indices]
        
        layer_row_means.append(sorted_means)
        layer_row_std_devs.append(sorted_std_devs)
        
    for tensor in weights_tensors:
        col_means = tensor.mean(dim=0).numpy()
        col_variances = tensor.var(dim=0).numpy()
        col_std_devs = np.sqrt(col_variances)
        sorted_indices = col_means.argsort()
        sorted_means = col_means[sorted_indices]
        sorted_std_devs = col_std_devs[sorted_indices]
        
        layer_col_means.append(sorted_means)
        layer_col_std_devs.append(sorted_std_devs)    

    fig_row = plt.figure(figsize=(12, 8))
    
    # Plot each layer's mean and std dev with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_layers))  # Generate colors
    for i in range(num_layers):
        plt.plot(layer_row_means[i], color=colors[i], label=f'Layer {i+1} Mean', linewidth=3)
        plt.fill_between(range(len(layer_row_means[i])),
                        layer_row_means[i] - layer_row_std_devs[i],
                        layer_row_means[i] + layer_row_std_devs[i],
                        color=colors[i], alpha=0.2)

    plt.xlabel('Row #')
    plt.ylabel('Values')
    plt.title(f'Mean Values and Standard Deviation of ROWS for {title}')
    plt.legend()
    plt.grid(True)
    
    
    fig_col = plt.figure(figsize=(16, 16))
    
    ax1 = fig_col.add_subplot(211)  # Top subplot
    ax2 = fig_col.add_subplot(212)  # Bottom subplot
    
    # Generate colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_layers))

    # Plot the data
    for i in range(num_layers):
        if i == 0:
            ax1.plot(layer_col_means[i], color=colors[i], label=f'Column {i+1} Mean', linewidth=3)
            ax1.fill_between(range(len(layer_col_means[i])),
                            layer_col_means[i] - layer_col_std_devs[i],
                            layer_col_means[i] + layer_col_std_devs[i],
                            color=colors[i], alpha=0.2)
            ax1.grid(True)
            ax1.legend()
        else:
            ax2.plot(layer_col_means[i], color=colors[i], label=f'Column {i+1} Mean', linewidth=3)
            ax2.fill_between(range(len(layer_col_means[i])),
                            layer_col_means[i] - layer_col_std_devs[i],
                            layer_col_means[i] + layer_col_std_devs[i],
                            color=colors[i], alpha=0.2)
            ax2.grid(True)
            ax2.legend()
    
    plt.tight_layout()
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