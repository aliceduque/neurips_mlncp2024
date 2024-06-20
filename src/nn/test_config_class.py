import torch
import matplotlib.pyplot as plt
from math import log10
from src.data.dataset_ops import get_test_loader
from src.nn.train_test_run import test_network
from src.graphic.plot_ops import make_plot
from src.utils.file_ops import save_variable, create_folder

class Test_Config:
    def __init__(self,
                 noise_range,
                 noise_points,
                 repetition,
                 data_file,
                 calc_acc,
                 calc_snr,
                 calc_entropy,
                 calc_gaussianity,
                 plot,
                 test_noises,
                 database,
                 device,
                 ):
        
        self.calc_acc = calc_acc
        self.calc_snr = calc_snr
        self.calc_entropy = calc_entropy
        self.calc_gaussianity = calc_gaussianity
        self.noise_points = noise_points
        self.repetition = repetition
        self.test_noises = test_noises
        self.noise_range = noise_range
        self.noise_types = len(test_noises)
        self.plot = plot
        self.database = database
        self.data_file = data_file
        self.device = device

        self.out = self.Out(self)

    class Out:
        def __init__ (self, t):
            if t.calc_acc:
                self.accuracy = torch.zeros((t.repetition, t.noise_points, t.noise_types))
            if t.calc_snr:
                self.snr = torch.zeros((t.repetition, t.noise_points, t.noise_types))
            if t.calc_entropy:
                self.entropy = torch.zeros((t.repetition, t.noise_points, t.noise_types))
            if t.calc_gaussianity:
                self.gaussianity = torch.zeros((t.repetition, t.noise_points, t.noise_types))
           
    
    # def test_loader(self, device):
    #     self.test_load = get_test_loader(database=self.database, root=self.data_file,
    #                                                         reduced=False, device=device)


    def test(self, model, rep, noise_value, noise_type):
        if self.calc_acc:
            self.out.accuracy[rep,noise_value,noise_type] = test_network(model, self.test_load)[1]
        else:
            self.out.accuracy = None

    def create_test_mat(self):
        test_mat = torch.zeros((self.noise_types,self.noise_points,4))
        noise_variance = torch.logspace(log10(self.noise_range[0]), log10(self.noise_range[1]), self.noise_points)   
        for k in range(self.noise_types):
            test_mat[k,:,k] = noise_variance
        return test_mat


    def plots(self, activation, train_vec, cwd):
        for att_name, att_value in self.out.__dict__.items():
            if att_value is not None:
                fig = make_plot(activation, att_name, att_value, train_vec,
                                self.noise_points, self.noise_range, self.test_noises)
                if self.plot=='save':
                    fig.savefig(rf'{att_name}/{torch.amax(train_vec):.1f}.png')
                    plt.close()
                if self.plot=='show':
                    plt.show()

    def save_points(self, activation, train_vec, cwd):
        for att_name, att_value in self.out.__dict__.items():
            if att_value is not None:
                    create_folder(cwd,att_name,cd=False)
                    save_variable(att_value, rf"{att_name}/{torch.amax(train_vec):.1f}.pkl")

    