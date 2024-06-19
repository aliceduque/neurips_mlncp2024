import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.functional

from src.utils.utils import *
from src.utils.file_ops import *
from src.graphic.plot_ops import weights_histogram
from src.nn.train_test_run import *
from src.nn.nn_classes import *
from src.nn.nn_operations import *
from src.base.dicts import *
from datetime import datetime
from src.nn.train_config_class import Train_Config
from src.nn.test_config_class import Test_Config





database = 'MNIST'
data_file = rf'C:\\Users\\220429111\\Box\\University\\PhD\\Codes\\Python\\neural_net_noise\\data\\'
device = 'cpu'
noise_on_activation = 'after'


train = Train_Config(
                    train_noise_types=['AddUnc', 'AllNoi'],
                    train_noise_values=[0.1],
                    activations = ['sigm','relu'],
                    baseline=False,                    
                    learning_rate=0.02,
                    num_epochs=2,
                    optimizer = 'Adam',
                    save_histogram=False,
                    save_parameters=False,
                    save_train_curve=False,
                    database=database,
                    data_file = data_file,
                    noise_on_activation = noise_on_activation                    
                    )

                    

test = Test_Config(
                    noise_range=[0.01, 10],
                    noise_points=2,
                    repetition=5,
                    test_noises=['AddUnc', 'AddCor', 'MulUnc', 'MulCor'],
                    calc_acc=True,
                    calc_entropy=False,
                    calc_gaussianity=False,
                    calc_snr=False,
                    plot ='save',
                    database=database,
                    data_file=data_file                    
                    )


train_mat = train.create_train_mat()
train.train_loader(device)
print('train_mat: ',train_mat)

test_mat = test.create_test_mat()
test.test_loader(device)
print('test_mat: ',test_mat)


root = "C:\\Users\\220429111\\Box\\University\\PhD\\Codes\\Python\\neural_net_noise\\outcomes\\MNIST\\20240618_test"


for a, activation in enumerate(train.activations):
    cwd = create_folder(root, activation, cd=True)
    for train_vec in train_mat:
        noise_type = noise_label(train_vec)
        print(noise_type)
        cwd = create_folder(cwd, noise_type, cd = True)
        # train.create_directories(cwd)
        model = create_net(activation, noise_on_activation, train_vec)
        print(rf'Activation: {activation} \\ Noise {noise_on_activation} activation \\ Noise_vector: {train_vec}')
        train.train_and_save(model, cwd, train_vec)
        for m,mat in enumerate(test_mat):
            for v, test_vec in enumerate(mat):
                assign_to_model(model, test_vec)
                for r in range(test.repetition):
                    test.test(model, database, data_file,r, v, m)
        test.save_points(activation, train_vec, cwd)
        test.plots(activation, train_vec, cwd)
        cwd = up_one_level(cwd)
    cwd = up_one_level(cwd)
train.save_config_to_file(root)






