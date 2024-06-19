# Overnight script
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from  src.base.init import *
from src.data.dataset_ops import get_test_loader
from src.nn.train_test_run import run_network
from src.nn.nn_classes import *
from src.utils.file_ops import save_model_parameters, load_model_parameters, save_variable, create_folder
from src.utils.utils import extract_initials


baseline = True


train = Train_Config(
    
)


test = Test_Config(
    activations = ['sigm', 'relu', 'relu_bound', 'relu_bound_symm', 'tanh'],
    calc_acc = True,
    calc_snr = True,
    calc_entropy = True,
    calc_gaussianity = True,
    noise_train_types = ['AddUnc', 'AddCor', 'MulUnc', 'MulCor', 'AllNoi', 'MulUnc_MulCor']
    noise_train_levels = [0.1, 0.5, 1.0, 5.0]
)

outputs = Outputs()


device ='cpu'
print(is_cuda)

learn_rates_dict = {'sigm': 0.01,
                     'relu': 0.005,
                     'relu_bound': 0.01,
                     'relu_bound_symm': 0.01,
                     'tanh': 0.008}


root = rf"C:\\Users\\220429111\\Box\\University\\PhD\\Codes\\Python\\MNIST\\results\\full_MNIST\\20240611_noisy_training\\batch_size_128"
training_epochs =70
activations = ['sigm', 'relu', 'relu_bound', 'relu_bound_symm', 'tanh']
colors = ['blue', 'green', 'red', 'orange']
labels = ['Additive Uncorrelated', 'Additive Correlated', 'Multiplicative Uncorrelated', 'Multiplicative Correlated', 'All Noises']
samples_per_batch = 7
noise_training = torch.tensor([0.1, 0.5, 1.0, 5.0])
# noise_training = torch.tensor([0.1])
test_loader = get_test_loader(reduced=False, device = device)
number_nets = 4
training_nets = 4

noise_variance = torch.logspace(np.log10(0.01), np.log10(10), 50)

noise_mat = torch.zeros((number_nets, len(noise_variance), 4))
training_mat =torch.zeros((training_nets+1, len(noise_training), 4))

for k,mat in enumerate(noise_mat):
    mat[:,k] = noise_variance

for k,mat in enumerate(training_mat):
    if k<number_nets:
        mat[:,k] = noise_training
    else:
        mat[:,:] = noise_training.view((len(noise_training),1))

# for k,mat in enumerate(training_mat):
#     mat[:,k+2] = noise_training

# for k,mat in enumerate(training_mat):
#     mat[:,k+2] = noise_training


print(training_mat)

def noise_label (vec):
    labels = ['Additive Uncorrelated', 'Additive Correlated', 'Multiplicative Uncorrelated', 'Multiplicative Correlated', 'All Noises']
    all_equal = torch.all(vec.eq(vec[0]))
    if all_equal:
        i = 4
    else:
        i = torch.argmax(vec)
    label = extract_initials(labels[i])
    return label


for a, activation in enumerate(activations):    # print('activation: ', activation)
    print('learning rate: ', lr)
    print('baseline')
    net = create_net(activation,[0,0,0,0]).to(device)
    ax = run_network(net, num_epochs=training_epochs, lr=lr, train=True, test=False,plot=True)
    ax.savefig(f'training_plot_baseline.png')
    parameter_path = f'param_BASELINE.pth'
    plt.close()
    save_model_parameters(net,parameter_path)
    ax = weights_histogram(net,f'Network {activation} noiseless (baseline)')
    ax.savefig(f'histogram_baseline.png')
    plt.close()
    acc = calc_accuracy(activation, parameter_path, noise_mat, test_loader, noise_variance, device)
    save_variable(acc,f'points_acc_baseline.pkl')
    ax = plot_traces_fill(noise_variance,acc,'Accuracy',f'Accuracy: {activation} trained without noise (baseline)')
    ax.savefig(f'acc_baseline.png')
    lr = learn_rates_dict[activation]
    for m, mat in enumerate(training_mat):
        cwd = create_folder(root,noise_label(mat[0]), cd=True)
        for vec in mat:
            print('activation: ', activation)
            print('learning rate: ', lr)
            print('noise type: ',noise_label(mat[0]))
            print('training with noise vector: ', vec)
            net = create_net(activation, vec).to(device)
            ax = run_network(net, num_epochs=training_epochs, lr=lr, train=True, test=False, plot=True)
            parameter_path = f'param_{noise_label(vec)}_{torch.amax(vec)}.pth'
            ax.savefig(f'training_plot_{noise_label(vec)}_{torch.amax(vec)}.png')
            plt.close()
            save_model_parameters(net,parameter_path)
            for n, noise in enumerate(noise_mat):
                for v, vec_eval in enumerate(noise):
                    net = create_net(activation, noise)
                    load_model_parameters(net, parameter_path)
                    acc



            ax = weights_histogram(net,f'Network {activation} trained with {noise_label(vec)} = {torch.amax(vec)}')
            ax.savefig(f'histogram_{noise_label(vec)}_{torch.amax(vec)}.png')
            plt.close()
            acc = calc_accuracy(activation, parameter_path, noise_mat, test_loader, noise_variance, device)
            save_variable(acc,f'points_acc_{noise_label(vec)}_{torch.amax(vec)}.pkl')
            ax = plot_traces_fill(noise_variance,acc,'Accuracy',f'Accuracy: {activation} trained with {noise_label(vec)} = {torch.amax(vec)}')
            ax.savefig(f'acc_{noise_label(vec)}_{torch.amax(vec)}.png')
            plt.close()
        os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = os.getcwd()

    os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))
    cwd = os.getcwd()











