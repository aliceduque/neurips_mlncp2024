import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_ops import load_variable
from src.base.dicts import label_dict
from src.graphic.plot_ops import *
tested_with = []

color_map = {'Bas': 'black', 'AddUnc': 'blue', 'AddCor': 'green', 'MulUnc': 'red', 'MulCor':'orange', 'AllNoi': 'purple'}
index_map = {'AddUnc': 0, 'AddCor': 1, 'MulUnc': 2, 'MulCor': 3}

batch_size = 128
noise1 = ['Bas', '0.0']
noise2 = ['Bas', '0.0']
tested_with.append (index_map.get('AddUnc'))
activation = 'sigm'


# root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240619_NORMALISED_noise_before_activation\{activation}"
root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240805_300_neurons\{activation}"
acc_1 = load_variable(rf"{root}\{noise1[0]}\accuracy\{noise1[1]}.pkl")
root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240822_reg_addunc\{activation}"
acc_2 = load_variable(rf"{root}\{noise2[0]}\\accuracy\{noise2[1]}.pkl")

print(acc_1.shape)
print(acc_2.shape)

noise1 = ['Bas', '0.0']
noise2 = ['Bas', '0.0']
tested_with .append(index_map.get('AddCor'))
root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240805_300_neurons\{activation}"
acc_3 = load_variable(rf"{root}\{noise1[0]}\accuracy\{noise1[1]}.pkl")
root = rf"C:\Users\220429111\Box\University\PhD\Codes\Python\neural_net_noise\outcomes\MNIST\20240822_reg_addcor\{activation}"
acc_4 = load_variable(rf"{root}\{noise2[0]}\accuracy\{noise2[1]}.pkl")


noise_variance = np.logspace(np.log10(0.001), np.log10(1), 100)
points = np.stack((acc_1,acc_2, acc_3, acc_4),axis=0)

labels = [rf'Trained w/ {noise1[0]} = {noise1[1]}', rf'Trained w/ {noise2[0]} = {noise2[1]}']
colors = [color_map.get(noise1[0]), color_map.get(noise2[0])]
colors = ['blue', 'blue', 'green', 'green']
labels = [ 'Uncorrelated (standard training)', 'Uncorrelated (noise-aware training)', 'Correlated (standard training)', 'Correlated (noise-aware training)']
noises = ['Additive uncorrelated', 'Additive correlated', 'Multiplicative uncorrelated', 'Multiplicative correlated']
linestyle = ['solid', 'dashed', 'solid', 'dashed']

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
    print(m)
    plt.plot(noise_variance, mean[:,tested_with[m//2]], color=colors[m], label = labels[m], linewidth=3, linestyle=linestyle[m])
    # plt.fill_between(noise_variance, np.mean(mat[tested_with], axis=0) - np.std(mat[tested_with], axis=0),
    #                   np.mean(mat[tested_with], axis=0) - np.std(mat[tested_with], axis=0), color=colors[m], alpha=0.3)
plt.xscale('log')
plt.title('Accuracy', fontsize=15)
plt.xticks(fontsize=14)  # Change x-axis tick size
plt.yticks(fontsize=14)
# plt.ylim([60, 100]) 
plt.grid(True)
# plt.title(f'{activation}: Accuracy under {noises[tested_with]} noise')

# plt.xlabel(f'{noises[tested_with]} noise variance')
plt.ylabel(f'Accuracy (%)')
plt.xlabel(f'Noise variance')

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