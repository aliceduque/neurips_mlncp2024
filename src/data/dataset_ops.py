# Dataset operations
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from ..base.init import SELECTED_CLASSES, BATCH_SIZE

selected_classes = SELECTED_CLASSES
normalization = transforms.Normalize((0.1305,), (0.3081,))
transformations = transforms.Compose([transforms.ToTensor(), normalization])

def get_train_loader(database, root, reduced=False, device='cpu'):
    rough_dataset = get_dataset(database, root)
    # if reduced:
    #   rough_indices = [i for i in range(len(rough_dataset)) if rough_dataset.targets[i] in selected_classes]
    #   rough_dataset = torch.utils.data.Subset(rough_dataset, rough_indices)

    img = rough_dataset.data.clone().detach().float().to(device)
    target = rough_dataset.targets.clone().detach().float().to(device)
    if reduced:
      reduced_indices = [i for i in range(len(target)) if target[i] in selected_classes]
      rough_tensor = TensorDataset(img[reduced_indices],target[reduced_indices])
    else:
       rough_tensor = TensorDataset(img,target)

    dataset_size = len(rough_tensor)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(rough_tensor, [train_size, val_size])

    train_loader = get_loader(train_dataset)
    validation_loader = get_loader(val_dataset)

    return train_loader, validation_loader


def get_test_loader(database, root, reduced=False, device='cpu'):
    print('test_loader device = ', device)
    test_dataset = get_dataset(database, root, train=False)
    img = test_dataset.data.clone().detach().float().to(device)
    target = test_dataset.targets.clone().detach().float().to(device)
    if reduced:
      reduced_indices = [i for i in range(len(target)) if target[i] in selected_classes]
      test_tensor = TensorDataset(img[reduced_indices],target[reduced_indices])
    else:
      test_tensor = TensorDataset(img,target)

    # if reduced:
    #   test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in selected_classes]
    #   test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = get_loader(test_tensor)
    return test_loader

def get_dataset(database, root="C:/Users/220429111/Box/University/PhD/Codes/Python/MNIST/data", train=True, transform=transformations,
                download=True):
    if database == 'MNIST':
      return datasets.MNIST(root=root, train=train, transform=transform,
                            download=download)

def get_single_sample (test_loader, sample=0):
  for images, labels in test_loader:
    image = images[sample]  # Extract the single image from the batch
    label = labels[sample]  # Extract the single label from the batch
    break  # Exit the loop after extracting the first sample
  return image, label


def extract_first_samples(test_loader, samples_per_batch=1):
  first_images = []
  first_labels = []
  batch_size = 10

  for batch in test_loader:
      for i in range(samples_per_batch):
        first_images.append(batch[0][i])
        first_labels.append(batch[1][i])

  i=0
  while (len(first_images) % batch_size != 0):
     first_images.append(batch[0][i])
     first_labels.append(batch[1][i])
     i+=1
     
  first_samples_dataset = TensorDataset(torch.stack(first_images), torch.stack(first_labels))
  new_test_loader = DataLoader(first_samples_dataset, batch_size=batch_size, shuffle=False)
  return new_test_loader


def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle)