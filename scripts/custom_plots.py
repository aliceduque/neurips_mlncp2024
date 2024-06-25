import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_ops import load_variable
from src.base.dicts import label_dict
from src.graphic.plot_ops import *

color_map = {'Bas': 'black', 'AddUnc': 'blue', 'AddCor': 'green', 'MulUnc': 'red', 'MulCor':'orange', 'AllNoi': 'purple'}
index_map = {'AddUnc': 0, 'AddCor': 1, 'MulUnc': 2, 'MulCor': 3}

batch_size = 128
noise1 = ['Bas', '0.0']
noise2 = ['MulUnc', '1.0']
tested_with = index_map.get('MulUnc')
activation = 'sigm'


# root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240619_NORMALISED_noise_before_activation\{activation}"
root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240621_noisy_training_after\{activation}"
acc_1 = load_variable(rf"{root}\\{noise1[0]}\\accuracy\\{noise1[1]}.pkl")
acc_2 = load_variable(rf"{root}\\{noise2[0]}\\accuracy\\{noise2[1]}.pkl")
print(acc_1.shape)
print(acc_2.shape)


noise_variance = np.logspace(np.log10(0.01), np.log10(10), 50)
points = np.stack((100*acc_1,100*acc_2),axis=0)

labels = [rf'Trained w/ {noise1[0]} = {noise1[1]}', rf'Trained w/ {noise2[0]} = {noise2[1]}']
colors = [color_map.get(noise1[0]), color_map.get(noise2[0])]

noises = ['Additive uncorrelated', 'Additive correlated', 'Multiplicative uncorrelated', 'Multiplicative correlated']

# old_colors = ['blue', 'green', 'red', 'orange']

# acc_1 = load_variable(rf"{root}\\{activation}\\learn_rate_0.003\\{noise1[0]}\\points_acc_{noise1[0]}_{noise1[1]}.pkl")
# acc_2 = load_variable(rf"{root}\\{activation}\\lr005mulcorr2.5mulunc2.5.pkl")


# plt.figure()
# for k in range(len(acc_1)):
#     plt.plot(noise_variance, np.mean(acc_1[k], axis=1), color = old_colors[k], linewidth=2, linestyle='dotted')
#     plt.plot(noise_variance, np.mean(acc_2[k], axis=1), color = old_colors[k], linewidth=3, label=noises[k])

fig = plt.figure()
for m,mat in enumerate(points):
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    plt.plot(noise_variance, mean[:,tested_with], color=colors[m], label = labels[m], linewidth=2)
    # plt.fill_between(noise_variance, np.mean(mat[tested_with], axis=0) - np.std(mat[tested_with], axis=0),
    #                   np.mean(mat[tested_with], axis=0) - np.std(mat[tested_with], axis=0), color=colors[m], alpha=0.3)
plt.xscale('log')
plt.title(f'{activation}: Accuracy under {noises[tested_with]} noise')
plt.xlabel(f'{noises[tested_with]} noise variance')


# fig = plt.figure()
# for k in range(len(points)):
#     plt.plot(noise_variance, np.mean(points[k,tested_with], axis=1), color = colors[k], label = labels[k], linewidth=2)
#     plt.fill_between(noise_variance, np.mean(points[k,tested_with], axis=1) - np.std(points[k, tested_with], axis=1),
#                       np.mean(points[k, tested_with], axis=1) + np.std(points[k, tested_with], axis=1), color=colors[k], alpha=0.3)
# plt.xscale('log')
# plt.title(f'{activation}: Accuracy under {noises[tested_with]} noise')
# plt.xlabel(f'{noises[tested_with]} noise variance')

# plt.title(f'{activation}: Trained w/ all noises combined vs trained w/ multiplicative noise only')
# plt.xlabel(f'Noise variance')
# plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()