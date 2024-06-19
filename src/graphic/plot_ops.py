# Plot functions

import matplotlib.pyplot as plt
import numpy as np
from src.base.dicts import *
from src.utils.utils import noise_label

def plot_loss_curve (num_epochs, training_losses, validation_accuracies):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(range(1, num_epochs + 1), training_losses, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:orange')
    ax2.plot(range(1, num_epochs + 1), validation_accuracies, label='Validation Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

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

    if ret:
        return fig
    else:
        plt.show()


def weights_histogram(net, title):
    weight_h1 = net.h1.weight.detach().numpy().flatten()
    weight_h2 = net.h2.weight.detach().numpy().flatten()
    weight_out = net.out.weight.detach().numpy().flatten()

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


def make_plot(activation, attribute, value, train_vec, noise_points, noise_range, test_noises):
    train_vec = train_vec.cpu().numpy()
    value = value.cpu().numpy()
    noise_vec = np.logspace(np.log10(noise_range[0]), np.log10(noise_range[1]), noise_points)    
    colors = [color_dict[v] for v in test_noises]
    labels = [label_dict[v] for v in test_noises]
    noise_labels = noise_label(train_vec)
    noise_value = np.amax(train_vec)
    title = (rf'{attribute}: {activation} trained with {noise_labels} = {noise_value:.1f}')
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
    return fig