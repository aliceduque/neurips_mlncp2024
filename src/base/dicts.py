from ..nn.nn_classes import *


create_net_dict = {
    'relu': (Net_ReLU, 0.005),
    'sigm': (Net_Sigm, 0.01),
    'relu_ln': (Net_ReLU_LN, 0.005),
    'relu_ln1': (Net_ReLU_LN1, 0.005),
    'relu_bound': (Net_ReLU_bound, 0.01),
    'relu_bound_symm': (Net_ReLU_bound_symm, 0.01),
    'leaky': (Net_Leaky, 0.01),
    'erf': (Net_Erf, 0.01),
    'gelu': (Net_GELU, 0.005),
    'tanh': (Net_Tanh, 0.05),
    'sigm_shift': (Net_Sigm_shift, 0.01)
}

learning_rates_dict = { 'relu': 0.005,
                        'sigm': 0.01,
                        'relu_ln':0.005,
                        'relu_ln1': 0.005,
                        'relu_bound': 0.01,
                        'relu_bound_symm': 0.01,
                        'leaky': 0.01,
                        'erf': 0.01,
                        'gelu': 0.005,
                        'tanh': 0.008,
                        'sigm_shift': 0.01
                    }


clean_text_dict = { 'sigm': 'Sigmoid',
                    'relu': 'ReLU',
                    'leak': 'Leaky ReLU',
                    'relu_ln': 'ReLU (Layer Norm)',
                    'relu_ln1': 'ReLU (1-layer norm)',
                    'relu_bound':'ReLU (bounded)',
                    'erf_lr02':'Erf',
                    'gelu': 'GeLU',
                    'tanh': 'Tanh'
                    }

color_dict = {'AddUnc': 'blue',
              'AddCor': 'green',
              'MulUnc': 'red',
              'MulCor': 'orange'}

label_dict = {'AddUnc': 'Additive Uncorrelated',
              'AddCor': 'Additive Correlated',
              'MulUnc': 'Multiplicative Uncorrelated',
              'MulCor': 'Multiplicative Correlated',
              'Bas': 'Baseline',
              'AddAll': 'Additive',
              'MulAll': 'Multiplicative'}

noise_to_index = {'AddUnc': 0,
                'AddCor': 1,
                'MulUnc': 2,
                'MulCor': 3}

noise_type_to_vector = {'AddUnc': [1,0,0,0],
                        'AddCor': [0,1,0,0],
                        'MulUnc': [0,0,1,0],
                        'MulCor': [0,0,0,1],
                        'AddAll': [1,1,0,0],
                        'MulAll': [0,0,1,1],
                        'AllNoi': [1,1,1,1],
                        'Bas': [0,0,0,0]}

gradients = {'h1.weight': [],
             'h2.weight': [],
             'out.weight': []}