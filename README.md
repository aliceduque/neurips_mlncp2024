# Improving Analog Neural Network Robustness: A Noise-Agnostic Approach with Explainable Regularizations

This project implements a framework that emulates the impact of hardware noise in analogue neural networks. It also explores the effects of both noise-aware training and the custom regularisation techniques introduced in our work.

Before running, please make sure to `pip install -r requirements.txt`

## Project Structure

- **noise-injection-nn/**
  - **codes/**
    - **scripts/**
        - `noisy_training.py`: Main script
    - **src/** : Holds all supporting functions and classes
        - **base/** : Contains basic definitions and dictionaries
        - **data/** : Contains basic database operations functions
        - **graphic/** : Contains all graphic functions (plots, histograms, etc)
        - **nn/** : Contains the core of neural networks (class definitions, training and testing)
        - **utils/** : Contains all-purpose supporting functions

## `noisy_training.py` in detail
To run this file, you must run in console `python -m scripts.noisy_training`

Analog noise is emulated by an operator that adds a random, Gaussian distributed signal on top of every activation.

Regarding neural network training, it can operate in various scenarios:
- Standard training (no noise injected);
- Noise-aware training (noise injected during training, type and intensity of noise adjustable);
- Training with proposed regularisation method (type and lambda values adjustable).

Regarding neural network testing, it can be evaluated under different noise profiles, with adjustable intensities:
- Additive noise (correlated and uncorrelated)
- Multiplicative noise (correlated and uncorrelated)

Other Functionalities:
- Built-in activation functions: Sigmoid,  Photonic sigmoid, ReLU (with and without layer norm), bounded ReLU, Leaky ReLU, GELU, Erf, Tanh;
- Can run on CPU or GPU;
- Can use previously obtained network parameters as a starting point for training a new network;
- Customisable width (parameter HIDDEN NEURONS in `init.py`)
- Noise can be modeled to be injected before or after activation

Limitations:
- Datasets supported: MNIST and Fashion MNIST
- Number of hidden layers is fixed to 2
- Regularisation for additive uncorrelated noise is deployed only for sigmoid and photonic sigmoid functions


## Dataset folder
After running for the first time, the desired dataset (options are `MNIST` or `FashionMNIST`) is downloaded to the folder `/data/`:
- **codes/**
    - **scripts/**
        - `noisy_training.py`: Main script
        - **data/**
            - **MNIST/**
            - **FashionMNIST/**

## Outcomes folder
For each new run of `noisy_training.py`, a folder is created, containing the relevant outcomes of the training/testing procedures, in the following path:
- **scripts/**
    - `noisy_training.py`: Main script
    - **outcomes/**
        - **MNIST/**
            - **20240913_test/** : user-defined name, with date index
                - **sigm/** : if multiple activation functions, multiple folders are created
                    - **Bas/** : noise type during training. If no noise, then Bas (=baseline)
                        - **accuracy/** : accuracy plots and points under several noise intensities (user-defined)
                        - **activations/** : activations distributions throughout the net
                        - **histogram/** : histograms of weights and biases (general and per-row)
                        - **parameters/** : parameters of the network after training (file .pth)
                        - **train_curve/** : plot of training loss, accuracy and validation accuracy versus epochs


Any queries, please feel free to reach out at alicedbo@gmail.com

