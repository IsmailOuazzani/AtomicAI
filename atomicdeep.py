# -*- coding: utf-8 -*-
"""atomicDeep.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFmKZZgEWAjlZHjw9YFOJ8wiHlMAMluT
"""

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GAME_PER_FILE = 1000
DATAPOINTS = 3000


class ChessDataset(Dataset):
    def __init__(self):
        self.length = DATAPOINTS

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = (idx // GAME_PER_FILE) * GAME_PER_FILE
        filename = f"dataset/{index}.npy"
        data = np.load(filename, mmap_mode="r")
        data_select = data[idx%GAME_PER_FILE, :, :, :]
        label = torch.tensor([(data_select[17,0,0])])
        return torch.from_numpy(np.copy(data_select[:17,:,:])).float(), label.float()



ds = ChessDataset()
ds.__getitem__(10000)[1]
print('label_data: ', ds.__getitem__(10000)[1])

def get_data_loader(batch_size):
  dataset = ChessDataset()
  total_size = len(dataset)
  train_size = int(0.6 * total_size)
  valid_size = int(0.2 * total_size)
  test_size = total_size - train_size - valid_size


  train_dataset, valid_dataset, test_dataset = random_split(
      dataset, [train_size, valid_size, test_size]
  )

  train_loader = DataLoader(
      train_dataset, batch_size= batch_size, shuffle=True
  )
  valid_loader = DataLoader(
      valid_dataset, batch_size = batch_size, shuffle=True
  )
  test_loader = DataLoader(
      test_dataset, batch_size = batch_size,  shuffle=True
  )
  return train_loader, valid_loader, test_loader

# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "outputs/model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.savefig("model_error.png".format(path))

    plt.show()
    # save the plot
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig("model_loss.png".format(path))

    plt.show()
    # save the plot

    plt.title("Train vs Validation Accuracy")
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("model_acc.png".format(path))

    plt.show()
    # save the plot

    
def eval(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_error = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            error = torch.abs(output - target).sum()
            total_error += error.item()
    n_samples = len(dataloader.dataset)
    avg_loss = total_loss / n_samples
    avg_error = total_error / n_samples
    model.train()
    return avg_loss, avg_error

def accuracy(net, loader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            predicted = torch.sign(outputs)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    net.train()
    return correct / total




def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader = get_data_loader(batch_size)
    ########################################################################
    criterion = nn.MSELoss()
    # optimizer = SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = Adam(net.parameters(), lr = learning_rate)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    num_epochs = num_epochs+1
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    epoch = 0
    train_err[epoch], train_loss[epoch] = eval(net, train_loader, criterion)
    train_acc[epoch] = accuracy(net, train_loader)
    val_err[epoch], val_loss[epoch] = eval(net, val_loader, criterion)
    val_acc[epoch] = accuracy(net, val_loader)
    print(("Before training, Epoch {}: Train err: {}, Train loss: {}, Train acc: {} |"+
            "Validation err: {}, Valid loss: {}, Validation acc: {}").format(
                epoch,
                train_err[epoch],
                train_loss[epoch],
                train_acc[epoch],
                val_err[epoch],
                val_loss[epoch],
                val_acc[epoch]))
    
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    print('start training')
    for it in range(num_epochs - 1):  # loop over the dataset multiple times
        epoch = it + 1
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
        #     if i % 100 == 0:
        #         print('epoch', epoch, 'batch', i)
            # Get the inputs
            inputs, labels = data
            # train on gpu if available
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.reshape(labels, (batch_size,1)) # TODO: switch batch_size to length of labels for uncomplete batches
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()    
       
        train_err[epoch], train_loss[epoch] = eval(net, train_loader, criterion)
        train_acc[epoch] = accuracy(net, train_loader)
        val_err[epoch], val_loss[epoch] = eval(net, val_loader, criterion)
        val_acc[epoch] = accuracy(net, val_loader)
        print(("Epoch {}: Train err: {}, Train loss: {}, Train acc: {} |"+
               "Validation err: {}, Valid loss: {}, Validation acc: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   train_acc[epoch],
                   val_err[epoch],
                   val_loss[epoch],
                   val_acc[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.name = "ChessNet"
        self.conv1 = nn.Conv2d(in_channels=17, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 2 * 2, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


neuralnet = ChessNet()
neuralnet.to(device)
batch_size = 100
learning_rate = 0.01
num_epochs = 20

# print number of trainable parameters
pytorch_total_params = sum(p.numel() for p in neuralnet.parameters() if p.requires_grad)
print("Number of trainable parameters: {}".format(pytorch_total_params))

train_net(neuralnet, num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate)

small_model_path = get_model_name(neuralnet.name, batch_size=batch_size, learning_rate=learning_rate, epoch=num_epochs)
plot_training_curve(small_model_path)

#!rm -rf /content/*