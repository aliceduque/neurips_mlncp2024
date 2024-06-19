# Overnight script

from  init import *
from requirements import *


def calc_accuracy(activation, filepath, noise_mat, test_loader):
    repetition = 2
    number_nets = 4
    acc = np.zeros((number_nets, len(noise_variance),repetition))
    net = []

    for k in range(number_nets):
        for i, vec in enumerate(noise_mat[k]):
            net = create_net(activation,vec)
            load_model_parameters(net, filepath)
            for r in range(repetition):
                acc[k][i][r] = 100*test_network(net, test_loader)[1]
    return acc


def calc_acc_averaging(activation, filepath, noise_mat, test_loader, N):
    repetition = 2
    number_nets = 4
    acc = np.zeros((number_nets, len(noise_variance),repetition))
    ratio = np.zeros_like(acc)
    net = []

    for k in range(number_nets):
        total = 0
        match = 0
        correct = 0
        net_clean = create_net(activation,np.zeros(4))
        load_model_parameters(net_clean, filepath)
        for i, vec in enumerate(noise_mat[k]):
                net = create_net(activation,vec)
                load_model_parameters(net, filepath)
                for r in range(repetition):
                    for ib, batch in enumerate(test_loader):
                        out_vec = np.zeros((N,BATCH_SIZE,OUTPUT_SIZE))
                        image, label = batch
                        answer = expand_expected_output(label)
                        output_clean = net_clean(image)
                        for n in range(N):
                            predicted_output = net(image)
                            out_vec[n,:,:] = predicted_output.detach().numpy()
                        match += (np.argmax(np.mean(out_vec, axis=0), axis=1) == np.argmax(output_clean.detach().numpy(),axis=1)).sum()
                        correct += (np.argmax(np.mean(out_vec, axis=0), axis=1) == np.argmax(answer.detach().numpy(),axis=1)).sum()
                        total += label.size(0)
                    ratio[k][i][r] = 100*match/total
                    acc[k][i][r] = 100*correct/total
    return acc,ratio


def calc_entropy(activation, filepath, noise_mat, test_loader):
    number_nets = 4
    ent = np.zeros((number_nets, len(noise_variance)))
    net = []
    for k in range(number_nets):
        for i, vec in enumerate(noise_mat[k]):
            net = create_net(activation,vec)
            load_model_parameters(net, filepath)
            ent[k][i] = test_network_entropy(net, test_loader)[2]
    return ent

def calc_entropy_misc(activation, filepath, noise_mat, test_loader):
    number_nets = 4
    ent_misc = np.zeros((number_nets, len(noise_variance)))
    net = []
    for k in range(number_nets):
        for i, vec in enumerate(noise_mat[k]):
            net = create_net(activation,vec)
            load_model_parameters(net, filepath)
            ent_misc[k][i] = test_network_misclassified(net, test_loader)[2]
    return ent_misc

def calc_snr(activation, filepath, noise_mat, test_loader):
    number_nets = 4
    net = []
    snr = np.zeros((number_nets,len(test_loader)))
    snr_mean = np.zeros((number_nets, len(noise_variance)))
    for k in range(number_nets):
        for i, vec in enumerate(noise_mat[k]):
            net = create_net(activation,vec)
            load_model_parameters(net, filepath)
            for ib, batch in enumerate(test_loader):
                image, label = batch
                snr[k][ib] = calculate_SNR(net, image, label)
            snr_mean[k][i] = snr[k].mean()
    return snr_mean

def calc_gaussianity(activation, filepath, noise_mat, test_loader):
    number_nets = 4
    net = []
    p = np.zeros((number_nets,len(test_loader)))
    p_mean = np.zeros((number_nets, len(noise_variance)))
    for k in range(number_nets):
        for i, vec in enumerate(noise_mat[k]):
            net = create_net(activation,vec)
            load_model_parameters(net, filepath)
            for ib, batch in enumerate(test_loader):
                image, label = batch
                p[k][ib] = calculate_gaussianity(net, image, label)
            p_mean[k][i] = p[k].mean()
    return p_mean

def weights_histogram(net, title):
    weight_h1 = net.h1.weight.detach().numpy().flatten()
    weight_h2 = net.h2.weight.detach().numpy().flatten()
    weight_out = net.out.weight.detach().numpy().flatten()

    #Plot histograms for the weight of each layer
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



root = r"C:/Users/220429111/Box/University/PhD/Codes/Python/MNIST/results/20240602_noisy_training"
training_epochs = 50
activations = ['relu_bound_symm']
colors = ['blue', 'green', 'red', 'orange']
labels = ['Additive Uncorrelated', 'Additive Correlated', 'Multiplicative Uncorrelated', 'Multiplicative Correlated', 'All Noises']
samples_per_batch = 7
noise_training = np.array([3.0])
learn_rates = [0.03]
test_loader = get_test_loader(reduced=True)
number_nets = 2
training_nets = 4
noise_variance = np.logspace(np.log10(0.01), np.log10(10), 50)

noise_mat = np.zeros((number_nets, len(noise_variance), 4))
training_mat = np.zeros((training_nets, len(noise_training), 4))

# for k,mat in enumerate(noise_mat):
#     mat[:,k] = noise_variance

# for k,mat in enumerate(training_mat):
#     if k<number_nets:
#         mat[:,k] = noise_training
#     else:
#         mat[:,:] = noise_training.reshape((len,1))

for k,mat in enumerate(training_mat):
    if k<number_nets:
        mat[:,k+2] = noise_training

print(training_mat)


for a, activation in enumerate(activations):
    print( 'activation: ', activation)
    cwd = create_folder(root,activation, cd=True)
    for l, lr in enumerate(learn_rates):
        cwd = create_folder(cwd,f'learn_rate_{lr}', cd=True)
        for m, mat in enumerate(training_mat):
            print(labels[m])
            cwd = create_folder(cwd,extract_initials(labels[m]), cd=True)
            for vec in mat:
                print('training with noise vector: ', vec)
                net = create_net(activation, vec)
                run_network(net, num_epochs=training_epochs, lr=lr, train=True)
                parameter_path = f'param_{extract_initials(labels[m])}_{np.amax(vec)}.pth'
                save_model_parameters(net,parameter_path)
                ax = weights_histogram(net,f'Network {activation} trained with {extract_initials(labels[m])} = {np.amax(vec)}')
                ax.savefig(f'histogram_{extract_initials(labels[m])}_{np.amax(vec)}.png')
                plt.close()
                acc = calc_accuracy(activation, parameter_path, noise_mat, test_loader)
                save_variable(acc,f'points_acc_{extract_initials(labels[m])}_{np.amax(vec)}.pkl')
                ax = plot_traces_fill(noise_variance,acc,'Accuracy',f'Accuracy: {activation} trained with {extract_initials(labels[m])} = {np.amax(vec)}')
                ax.savefig(f'acc_{extract_initials(labels[m])}_{np.amax(vec)}.png')
                plt.close()
            os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))
            cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = os.getcwd()
    os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))
    cwd = os.getcwd()











