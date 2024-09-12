# File operations
import torch
import os
import pickle


def create_folder(root,filepath_from_root, cd=False):
    root = root.replace('\\', '/')
    filepath = os.path.join(root,filepath_from_root)
    if not os.path.exists(filepath):
      os.makedirs(filepath)
    if cd:
      os.chdir(filepath)
    return os.getcwd()

def save_variable(variable, filepath):
  with open(filepath, 'wb') as f:
    pickle.dump((variable), f)

def load_variable(filepath):
  with open(filepath, 'rb') as f:
    variable = pickle.load(f)
  return variable

def save_model_parameters(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model_parameters(model, filepath):
    model.load_state_dict(torch.load(filepath))

def up_one_level(cwd):
    os.chdir(os.path.abspath(os.path.join(cwd, os.pardir)))
    cwd = os.getcwd()
    return cwd


