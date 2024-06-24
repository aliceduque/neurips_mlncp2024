import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.functional
from datetime import datetime
from multiprocessing import freeze_support

from src.utils.utils import *
from src.utils.file_ops import *
from src.graphic.plot_ops import weights_histogram
from src.nn.train_test_run import *
from src.nn.nn_classes import *
from src.nn.nn_operations import *
from src.base.dicts import *

from src.nn.train_config_class import Train_Config
from src.nn.test_config_class import Test_Config


def main():

    database = 'MNIST'
    data_file = rf'C:\\Users\\220429111\\Box\\University\\PhD\\Codes\\Python\\neural_net_noise\\data\\'
    device = 'cpu'
    noise_on_activation ='after'
    root = "C:\\Users\\220429111\\Box\\University\\PhD\\Codes\\Python\\neural_net_noise\\outcomes\\MNIST\\20240620_GPU_test"

    train = Train_Config(
                        train_noise_types = ['AddCor'],
                        train_noise_values = [0.1],
                        activations = ['relu'],
                        baseline =True,                    
                        learning_rate = 'specific',
                        num_epochs = 10,
                        optimizer = 'Adagrad',
                        save_histogram = True,
                        save_parameters = True,
                        save_train_curve = True,
                        save_gradients= False,
                        database=database,
                        data_file = data_file,
                        noise_on_activation = noise_on_activation,
                        device = device                    
                        )

                        

    test = Test_Config(
                        noise_range = [5, 10],
                        noise_points = 10,
                        repetition = 1,
                        test_noises = ['AddUnc','AddCor', 'MulUnc', 'MulCor'],
                        calc_acc = True,
                        calc_entropy = True,
                        calc_gaussianity = True,
                        calc_snr = True,
                        plot ='save',
                        database = database,
                        data_file = data_file,
                        device = device                    
                        )


    train_mat = train.create_train_mat()
    # train.train_loader(device)
    train.train_load, train.validation_load = get_train_loader(database, data_file, device=device) 

    test_mat = test.create_test_mat()
    test.test_load = get_test_loader(database=database, root=data_file, device=device)
    # test.test_loader(device)


    learning_rates_regime = train.learning_rate


    train.save_config_to_file(root)


    for a, activation in enumerate(train.activations):
        if learning_rates_regime == 'specific':
            train.learning_rate = learning_rates_dict[activation]
        cwd = create_folder(root, activation, cd=True)
        for train_vec in train_mat:
            noise_type = noise_label(train_vec)
            cwd = create_folder(cwd, noise_type, cd = True)
            model = create_net(activation, noise_on_activation, train_vec).to(device)
            print(rf'Activation: {activation} \\ Noise {noise_type} = {torch.amax(train_vec)} {noise_on_activation} activation')
            train.train_and_save(model, cwd, train_vec)
            for m,mat in enumerate(test_mat):
                for v, test_vec in enumerate(mat): 
                    print(test_vec)
                    model = assign_to_model(model, test_vec)
                    for r in range(test.repetition):
                        test.test(model, r, v, m)
            test.save_points(activation, train_vec, cwd)
            test.plots(activation, train_vec, cwd)
            cwd = up_one_level(cwd)
        cwd = up_one_level(cwd)



if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()





