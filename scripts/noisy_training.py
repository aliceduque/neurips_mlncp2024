import torch
from datetime import datetime
from src.nn.test_config_class import Test_Config
from src.nn.train_config_class import Train_Config
from src.nn.nn_operations import create_net, assign_to_model, discretise_weights
from src.data.dataset_ops import get_train_loader, get_test_loader
from src.utils.file_ops import create_folder, load_model_parameters, up_one_level
from src.utils.utils import noise_label
from src.base.dicts import create_net_dict



def main():
    test_name = 'test_with_none'
    database = 'MNIST'
    device = 'cpu'
    noise_on_activation = 'after'
    baseline_initialisation = True
    noise_no_grad = False


  
    data_file = rf'C:/Users/220429111/Box/University/PhD/Codes/Python/neural_net_noise/data'
    train = Train_Config(
        train_noise_types = ['AddUnc'],
        train_noise_values = [0.0],
        activations = ['sigm'],
        baseline = True,
        learning_rate ='specific',
        num_epochs = 50,
        optimizer = 'adam',
        regularisation = 'h2_saturation_out_l2',
        lambda_reg = 1e-2,
        reg_config = [0.5, 0.01, 1], # Saturation of h2, L2 of h2, L2 out
        save_histogram = True,
        save_parameters = True,
        save_train_curve = True,
        save_gradients = False,
        database = database,
        data_file = data_file,
        noise_on_activation = noise_on_activation,
        device = device,

    )

    test = Test_Config(
        noise_range = [0.001, 1],
        noise_points = 30,
        repetition = 3,
        test_noises = ['AddUnc', 'AddCor', 'MulUnc', 'MulCor'],
        calc_acc = True,
        calc_entropy = False,
        calc_gaussianity = False,
        calc_snr = False,
        plot = 'save',
        database = database,
        data_file = data_file,
        device = device
    )
    
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d")    
    data_file = rf'C:/Users/220429111/Box/University/PhD/Codes/Python/neural_net_noise/data'
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
            print(train_vec)
            if baseline_initialisation:
                if not torch.all(torch.eq(train_vec, torch.tensor([0., 0., 0., 0.]))):
                    load_model_parameters(model,rf"{root}/{activation}/Bas/parameters/0.00.pth")
                else:
                    load_model_parameters(model,rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240805_300_neurons\{activation}\Bas\parameters\0.00.pth")    
                print('loaded baseline parameters')
            print(rf'Activation: {activation} \\ Noise {noise_type} = {torch.amax(train_vec)} {noise_on_activation} activation')
            train.train_and_save(model, cwd, train_vec)
                    
            for m, mat in enumerate(test_mat):
                for v, test_vec in enumerate(mat):
                    print(test_vec)
                    model = assign_to_model(model, test_vec)
                    for r in range(test.repetition):
                        test.test(model, r, v, m)
            test.save_points(activation, train_vec, cwd)
            test.plots(activation, train_vec, noise_on_activation)
            cwd = up_one_level(cwd)
        cwd = up_one_level(cwd)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
