# Simplex operations
import numpy as np
import matplotlib as plt
import torch
from ..utils.utils import obtain_index
from ..base.init import SELECTED_CLASSES

VERTICES = np.array([[-np.sqrt(2)/2, 0], [np.sqrt(2)/2, 0], [0, np.sqrt(3/2)]])
M = np.array([[-np.sqrt(2)/2, 0], [np.sqrt(2)/2, 0], [0, np.sqrt(1.5)]])
selected_classes = SELECTED_CLASSES

def create_simplex_divisions():
  vertices = np.array([[-np.sqrt(2)/2, 0], [np.sqrt(2)/2, 0], [0, np.sqrt(3/2)]])
  fig, ax = plt.subplots()
  triangle = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
  ax.add_patch(triangle)

  ax.set_xlim(-np.sqrt(2)/2 - 0.1, np.sqrt(2)/2 + 0.1)
  ax.set_ylim(-0.1, np.sqrt(3/2) + 0.1)

  ax.set_aspect('equal')

  plt.text(-np.sqrt(2)/2 -0.2, -0.05, f'$P(x={selected_classes[0]})$', verticalalignment='top', fontsize=12,horizontalalignment='left')
  plt.text(np.sqrt(2)/2 -0.02 , -0.05, f'$P(x={selected_classes[1]})$', verticalalignment='top', fontsize=12)
  plt.text(0, np.sqrt(3/2)+0.02, f'$P(x={selected_classes[2]})$', verticalalignment='bottom', horizontalalignment='center', fontsize=12)

  endpoints = np.array([[0,0], [-np.sqrt(2)/4, np.sqrt(6)/4], [np.sqrt(2)/4, np.sqrt(6)/4]])
  for i in range(len(endpoints)):
    ax.plot([0, endpoints[i, 0]], [np.sqrt(1.5)/3, endpoints[i, 1]], color='grey', linewidth=0.8)

  # Hide the grid
  ax.grid(False)

  # Remove the frame
  ax.set_frame_on(False)

  # Remove the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  return(ax,fig)


def create_simplex_shaded():

  vertices = np.array([[-np.sqrt(2)/2, 0], [np.sqrt(2)/2, 0], [0, np.sqrt(3/2)]])
  fig, ax = plt.subplots()
  triangle = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
  ax.add_patch(triangle)

  ax.set_xlim(-np.sqrt(2)/2 - 0.1, np.sqrt(2)/2 + 0.1)
  ax.set_ylim(-0.1, np.sqrt(3/2) + 0.1)
  ax.set_aspect('equal')

  plt.text(-np.sqrt(2)/2 -0.2, -0.05, f'$P(x={selected_classes[0]})$', verticalalignment='top', fontsize=12,horizontalalignment='left')
  plt.text(np.sqrt(2)/2 -0.02 , -0.05, f'$P(x={selected_classes[1]})$', verticalalignment='top', fontsize=12)
  plt.text(0, np.sqrt(3/2)+0.02, f'$P(x={selected_classes[2]})$', verticalalignment='bottom', horizontalalignment='center', fontsize=12)

  green_points = np.array([[0, 0], [0, np.sqrt(1.5)/3], [np.sqrt(2)/4, np.sqrt(6)/4], [np.sqrt(2)/2, 0]])
  yellow_points = np.array([[0, 0], [0, np.sqrt(1.5)/3], [-np.sqrt(2)/4, np.sqrt(6)/4], [-np.sqrt(2)/2, 0]])
  red_points = np.array([[0, np.sqrt(1.5)], [-np.sqrt(2)/4, np.sqrt(6)/4], [0, np.sqrt(1.5)/3], [np.sqrt(2)/4, np.sqrt(6)/4]])

  ax.fill(green_points[:, 0], green_points[:, 1], color='green', alpha=0.2)
  ax.fill(yellow_points[:, 0], yellow_points[:, 1], color='orange', alpha=0.2)
  ax.fill(red_points[:, 0], red_points[:, 1], color='brown', alpha=0.2)

  ax.grid(False)
  ax.set_frame_on(False)
  ax.set_xticks([])
  ax.set_yticks([])

  return(ax,fig)


def plot_simplex(probabilities, expected_class, fig, baseline = False, average = False):
  color = ['orange', 'green', 'brown']
  # Transformation matrix
  probabilities = probabilities.cpu().numpy()
  M = np.array([[-np.sqrt(2)/2, 0], [np.sqrt(2)/2, 0], [0, np.sqrt(1.5)]])
  if torch.is_tensor(expected_class):
    for i in range(probabilities.shape[0]):  # Iterate over rows
      out = probabilities[i].T @ M
      cls = obtain_index(expected_class[i])
      if baseline:
         fig.scatter(out[0], out[1], color='black', s =35, marker='o')
      elif average:
         fig.scatter(out[0], out[1], color='black', s =35, marker='X')
      else:
        fig.scatter(out[0], out[1], color=color[cls], s =10)
      
  
  else:
      out = probabilities.T @ M
      cls = obtain_index(expected_class)
      if baseline:
         fig.scatter(out[0], out[1], color='black', s =35, marker='o')
      elif average:
         fig.scatter(out[0], out[1], color='black', s =35, marker='x')
      else:      
        fig.scatter(out[0], out[1], c=color[cls], s=10)

  return fig


def plot_point(ax, vec, color='gray'):
    M = np.array([[-np.sqrt(2)/2, 0], [np.sqrt(2)/2, 0], [0, np.sqrt(1.5)]])
    out = vec.T @ M
    ax.scatter(out[0], out[1], color=color, s=12)