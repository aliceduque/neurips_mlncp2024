# Train and test network

import torch
import torch.nn as nn
from ..data.dataset_ops import get_test_loader, get_train_loader
from ..base.init import NUM_EPOCHS,LEARNING_RATE,INPUT_SIZE,OUTPUT_SIZE
from ..graphic.plot_ops import plot_loss_curve
from ..graphic.simplex_ops import create_simplex_shaded, plot_simplex
from ..utils.utils import get_actual_output, expand_expected_output

def run_network(net, database, root, num_epochs=NUM_EPOCHS, train=True, test=True, lr=LEARNING_RATE, reduced=False, plot=False):
    dev = next(net.parameters()).device
    if train:
      train_loader, validation_loader = get_train_loader(database=database, root=root, reduced=reduced, device=dev)
      cross_entropy_loss = nn.CrossEntropyLoss()
      cross_entropy_loss.to(dev)
      loss_function = create_loss_function(cross_entropy_loss)
      optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
      print('learning rate: ',lr)
      ax = train_network(net, train_loader, num_epochs, loss_function, optimizer, validation_loader, plot=plot)
    if test:
      test_loader = get_test_loader(database=database, root=root, reduced=reduced, device=dev)
      test_network(net, test_loader)
    return ax


def train_network(model, train_loader, num_epochs, loss_function, optimizer, validation_loader, plot=False):
    dev = next(model.parameters()).device
    training_losses = []
    validation_accuracies = []
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, expected_outputs) in enumerate(train_loader):
            images, expected_outputs = images.to(dev, non_blocking=True), expected_outputs.to(dev, non_blocking=True)
            outputs = model(images)
            loss = loss_function(outputs, expected_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        avg_training_loss = epoch_loss / num_batches
        training_losses.append(avg_training_loss)
        _, validation_accuracy, _ = test_network(model, validation_loader)
        validation_accuracies.append(validation_accuracy * 100) 
        print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(
            epoch + 1, num_epochs, avg_training_loss, validation_accuracy * 100))
   
    if plot:
        fig = plot_loss_curve(num_epochs,training_losses,validation_accuracies)
    else:
       fig = None

    return fig


def test_network(model, data_loader, simplex=False, epoch_info=""):
    dev = next(model.parameters()).device
    model = model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        if simplex:
          ax, fig = create_simplex_shaded()
        for batch in data_loader:
            images, expected_outputs = batch
            images, expected_outputs = images.to(dev, non_blocking=True), expected_outputs.to(dev, non_blocking=True)
            outputs = model(images)
            expected_classes = expected_outputs.type(torch.int)
            # get the predicted value from each output in the batch
            predicted_index = torch.argmax(outputs, dim=1).type(torch.int)
            predicted_outputs = get_actual_output(predicted_index)
            if simplex:
              ax = plot_simplex(outputs,expected_classes,ax)
            else:
              ax = None
            correct += (predicted_outputs == expected_outputs).sum()
            total += expected_outputs.size(0)

        results_str = f"Test data results: {float(correct)/total}"
        if epoch_info:
            results_str += f", {epoch_info}"
        #print(results_str)
        # if simplex:
        #   plt.show()
        print(float(correct)/total)
    return batch, float(correct)/total, ax




def create_loss_function(loss_function, output_size=OUTPUT_SIZE):
    def calc_loss(outputs, target):
        targets = expand_expected_output(target, output_size)
        return loss_function(outputs, targets)
    return calc_loss