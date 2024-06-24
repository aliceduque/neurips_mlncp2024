
import torch
import matplotlib.pyplot as plt
from src.base.dicts import *
from src.data.dataset_ops import get_train_loader
from src.nn.train_test_run import create_loss_function, train_network
from src.utils.file_ops import create_folder, save_model_parameters
from src.utils.utils import noise_label
from src.graphic.plot_ops import weights_histogram
from datetime import datetime

class Train_Config:
    def __init__(self,
                 train_noise_types,
                 train_noise_values,
                 activations,
                 database,
                 data_file,                
                 baseline,
                 learning_rate,
                 num_epochs,
                 optimizer,
                 noise_on_activation,                 
                 save_train_curve,
                 save_histogram,
                 save_parameters,
                 save_gradients,
                 device
                 ):
        self.train_noise_types = train_noise_types
        self.train_noise_values = train_noise_values
        self.baseline = baseline
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_train_curve = save_train_curve
        self.save_histogram = save_histogram
        self.save_parameters = save_parameters
        self.activations = activations
        self.database = database
        self.data_file = data_file
        self.optimizer = optimizer
        self.noise_on_activation = noise_on_activation
        self.device = device
        self.save_gradients = save_gradients


    def create_train_mat (self):
        rep_vec = torch.tensor(self.train_noise_values).repeat(len(self.train_noise_types))
        full_mat = rep_vec.repeat_interleave(4).reshape(-1, 4)
        mask_mat = torch.zeros((len(self.train_noise_types) * len(self.train_noise_values), 4))
        noise_indexes = torch.tensor([noise_type_to_vector[v] for v in self.train_noise_types])
        for v in range(len(mask_mat)):
            mask_mat[v] = noise_indexes[v // len(self.train_noise_values)]
        train_mat = full_mat*mask_mat
        if self.baseline:
            baseline_row = torch.tensor([0, 0, 0, 0]).unsqueeze(0)
            train_mat = torch.cat((baseline_row, train_mat), dim=0)
        return train_mat

    # def train_loader(self, device):
    #     self.train_load, self.validation_load = get_train_loader(database=self.database, root=self.data_file,
    #                                                                  reduced=False, device=device)

    def define_optimizer(self, model, opt):
        if opt == 'Adagrad' or 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)
        elif opt == 'Adam' or 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif opt == 'SGD' or 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        elif opt == 'Adadelta' or 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"I don't know that optimizer {opt}! Please include it in define_optimizer function")
        return optimizer


    def train_and_save(self, model, root, train_vec):
        dev = next(model.parameters()).device
        cross_entropy_loss = nn.CrossEntropyLoss()
        cross_entropy_loss.to(dev)
        loss_function = create_loss_function(cross_entropy_loss)
        optimizer = self.define_optimizer(model,self.optimizer)
        fig1, fig2 = train_network(model, self.train_load, self.num_epochs, loss_function, optimizer, 
                            self.validation_load, plot_curve=self.save_train_curve, plot_gradient=self.save_gradients)

        if self.save_train_curve:
            cwd = create_folder(root, 'train_curve', cd=False)
            fig1.savefig(rf"train_curve/{torch.amax(train_vec):.2f}.png")
            print('saved plot curve')
            plt.close()
        if self.save_gradients:
            cwd = create_folder(root, 'gradients', cd=False)
            fig2.savefig(rf"gradients/{torch.amax(train_vec):.2f}.png")
            print('saved gradients')
            plt.close()
        if self.save_histogram:
            cwd = create_folder(root, 'histogram', cd=False)
            fig = weights_histogram(model,f'Network trained with {noise_label(train_vec)} = {torch.amax(train_vec):.2f}')
            plt.savefig(rf"histogram/{torch.amax(train_vec):.2f}.png")
            print('saved histogram')
            plt.close()
        if self.save_parameters:
            cwd = create_folder(root, 'parameters', cd=False) 
            save_model_parameters(model,rf"parameters/{torch.amax(train_vec):.2f}.pth")

    def save_config_to_file(self, file_path):
        current_date = datetime.now()
        date_string = current_date.strftime("%Y%m%d")
        with open(rf"{file_path}/{date_string}_config_file.txt", 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("Learning rates dictionary: \n")
            for key, value in learning_rates_dict.items():
                f.write(f"{key}: {value}\n")


