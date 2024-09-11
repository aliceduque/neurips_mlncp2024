import torch
import torch.nn as nn
from datetime import datetime
from src.nn.test_config_class import Test_Config
from src.nn.train_config_class import Train_Config
from src.nn.nn_operations import create_net, assign_to_model
from src.data.dataset_ops import get_train_loader, get_test_loader
from src.utils.file_ops import create_folder, load_model_parameters, up_one_level
from src.graphic.plot_ops import activations_plot
from src.utils.utils import noise_label
from src.base.dicts import create_net_dict


def main():
    test_name = 'photonic_sigmoid_reg_sgd_120ep' # File name to contain results
    database = 'MNIST'
    device = 'cpu'
    noise_on_activation = 'after' # Noise injected after activation ('before' also possible)
    baseline_initialisation = True # If true Network will start off from a previously obtained set of weights and biases (baseline)
    noise_no_grad = False # If True, noise injections do not count towards autograd backpropagation (only relevant in multiplicative noise)
    activation_reg = 'sigm' # Sigmoid activation function
    data_file = rf'C:/Users/220429111/Box/University/PhD/Codes/Python/neural_net_noise/data'

  
    train = Train_Config(
        train_noise_types = [], # Noise types to be injected during noise-aware training ('AddUnc', 'AddCor', 'MulUnc', 'MulCor'). Must be empty for regularisation.
        train_noise_values = [], # Variance of injected noise during training
        activations = [activation_reg],
        baseline = True, # If True, network will be trained without noise before being trained with noise
        learning_rate = 5e-6, # If 'specific', then adapts to whichever activation function is in use, based on a dictionary
        num_epochs = 120,
        optimizer = 'adam',
        regularisation = f'addunc_{activation_reg}', # Regularisation type. For additive correlated, 'custom_sum', for uncorrelated, 'addunc_{activation_reg}'
        lambda_reg = 6e-3, # General lambda for regularisation
        reg_config = [0.01, 0.015, 3.0], # Applies to Additive Uncorrelated: [saturation of h2, L2 regularisation of h2, L2 regularisation out layer]
        save_histogram = True,
        save_parameters = True,
        save_train_curve = True,
        save_gradients = False, # Save gradients evolution: this takes a lot of time to compute, best left False as default
        database = database,
        data_file = data_file,
        noise_on_activation = noise_on_activation,
        device = device,

    )

    test = Test_Config(
        noise_range = [0.001, 1],
        noise_points = 40,
        repetition = 3, # Accuracy is measured as an average, this is the number of repetitions
        test_noises = ['AddUnc', 'AddCor', 'MulUnc', 'MulCor'],
        calc_acc = True,
        plot = 'save',
        database = database,
        data_file = data_file,
        device = device
    )
    
    
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d")    
    root = create_folder(rf"C:/Users/220429111/Box/University/PhD/Codes/Python/neural_net_noise/outcomes/{database}",
                         rf"{date_string}_{test_name}", cd = True)
    train_mat = train.create_train_mat()
    train.train_load, train.validation_load = get_train_loader(database, data_file, device = device)
    test_mat = test.create_test_mat()
    test.test_load = get_test_loader(database = database, root = data_file, device = device)
    learning_rates_regime = train.learning_rate
    train.save_config_to_file(root)

    for a, activation in enumerate(train.activations):
        if learning_rates_regime == 'specific':
            train.learning_rate = create_net_dict[activation][1]
        cwd = create_folder(root, activation, cd = True)
        for train_vec in train_mat:
            noise_type = noise_label(train_vec)
            cwd = create_folder(cwd, noise_type, cd = True)
            model = create_net(activation, noise_on_activation, train_vec, noise_no_grad).to(device)
            # init_weights_normal(model)
            print(train_vec)
            if baseline_initialisation:
                if not torch.all(torch.eq(train_vec, torch.tensor([0., 0., 0., 0.]))):
                    load_model_parameters(model,rf"{root}/{activation}/Bas/parameters/0.00.pth")
                    print('loaded baseline parameters')
                elif train.reg_type is not None:
                    load_model_parameters(model,rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240829_photonic_sigmoid_reg_from_scratch_120ep\{activation}\Bas\parameters\0.00.pth")    
                    print('loaded baseline parameters')
            print(rf'Activation: {activation} \\ Noise {noise_type} = {torch.amax(train_vec)} {noise_on_activation} activation')
            train.train_and_save(model, cwd, train_vec)
            cwd = create_folder(cwd, 'activations', cd=True)
            activations_plot(model, test.test_load)
            cwd = up_one_level(cwd)                
            for m, mat in enumerate(test_mat):
                for v, test_vec in enumerate(mat):
                    print(test_vec)
                    model = assign_to_model(model, test_vec)
                    for r in range(test.repetition):
                        test.test(model, r, v, m)
            test.save_points(activation, train_vec, cwd)
            test.plots(activation, train_vec, noise_on_activation)

        cwd = up_one_level(cwd)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
