# Utils
import torch
from ..base.init import SELECTED_CLASSES, OUTPUT_SIZE
from ..base.dicts import *

selected_classes = SELECTED_CLASSES

def extract_initials(input_string):
    words = input_string.split()  # Split the input string into words
    first_three_chars = [word[:3] for word in words]  # Get the first 3 characters of each word
    result = ''.join(first_three_chars)  # Concatenate the first 3 characters of each word
    return result

def expand_single_output(expected_output, output_size):
    x = [0.0 for _ in range(output_size)]
    # x[expected_output-3] = 1.0
    for i in range(output_size):
      if (expected_output == selected_classes[i]):
        x[i] = 1.0
    return x

def expand_expected_output(tensor_of_expected_outputs, output_size=OUTPUT_SIZE):
    dev = tensor_of_expected_outputs.device
    return torch.tensor([expand_single_output(expected_output.item(),
                                              output_size)
                         for expected_output in tensor_of_expected_outputs], device=dev)

def obtain_index(cls, output_size=OUTPUT_SIZE):
  for i in range(output_size):
    if cls == selected_classes[i]:
      index = i
  return index

def get_actual_output (vector_of_indices):
  dev = vector_of_indices.device
  x = torch.zeros_like((vector_of_indices),device=dev)
  for i in range(len(x)):
    x[i] = selected_classes[vector_of_indices[i]]+0.0
  return x

def noise_label (vec):
    if not isinstance(vec, torch.Tensor):
      vec = torch.tensor(vec)
    if torch.amax(vec) == 0:
      norm_vec = vec
    else:
      norm_vec = vec / torch.amax(vec)
    vector_to_noise_type = {tuple(v): k for k, v in noise_type_to_vector.items()}
    return vector_to_noise_type.get(tuple(norm_vec.tolist()), "Unknown vector")
    # print(vector_to_noise_type)

    # if not isinstance(vec, torch.Tensor):
    #     vec = torch.tensor(vec)
    # labels = ['Additive Uncorrelated', 'Additive Correlated', 'Multiplicative Uncorrelated', 'Multiplicative Correlated', 'All Noises', 'Baseline']
    # all_equal = torch.all(vec.eq(vec[0]))
    # if all_equal:
    #     if vec[0] == 0:
    #         i = 5
    #     else:
    #         i = 4
    # else:
    #     i = torch.argmax(vec)
    # label = extract_initials(labels[i])
    # return vector_to_noise_type

def discretise_tensor(tensor):
    rows, cols = tensor.size()
    discretised_tensor = torch.zeros_like(tensor)
    
    for i in range(rows):
        
        row = tensor[i]
        mean = row.mean()
        row = row-mean
        std = row.std()


        # value_list = torch.tensor([-2*std, -std, mean, std, 2*std], device=tensor.device)
        value_list = torch.tensor([-1.5*std, mean, 1.5*std], device=tensor.device)        
        distance = torch.abs(row.unsqueeze(-1) - value_list)
        min_indices = torch.argmin(distance, dim=-1)
        discretised_row = value_list[min_indices]
        discretised_row = discretised_row - mean
        discretised_tensor[i] = discretised_row

        
    
    return discretised_tensor

def extract_elements(input_list):
    if input_list and isinstance(input_list[0], list):
        return [item for sublist in input_list for item in sublist]
    else:
        return input_list