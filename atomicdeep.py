# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time
import torch.nn.functional as F
import os
import logging
import multiprocessing
import torchsummary


# Modify when changing dataset
GAME_PER_FILE = 5000
DATAPOINTS = 3000000


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
        # replace -1.0 with 0.0 in label, elementwise
        label = torch.where(label == -1.0, torch.zeros_like(label), label)
        return torch.from_numpy(np.copy(data_select[:17,:,:])).float(), label.float()



def get_data_loader(batch_size):
  dataset = ChessDataset()
  total_size = len(dataset)
  train_size = int(0.6 * total_size)
  valid_size = int(0.2 * total_size)
  test_size = total_size - train_size - valid_size


  train_dataset, valid_dataset, test_dataset = random_split(
      dataset, [train_size, valid_size, test_size]
  )

  # figure out the num workers for the data loader

  train_loader = DataLoader(
      train_dataset, batch_size= batch_size, shuffle=True, num_workers=4
  )
  valid_loader = DataLoader(
      valid_dataset, batch_size = batch_size, shuffle=True, num_workers=4
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
    plt.clf()
    
    # save the plot
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig("model_loss.png".format(path))
    plt.clf()
    # save the plot

    plt.title("Train vs Validation Accuracy")
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("model_acc.png".format(path))
    plt.clf()
    # save the plot

    
def eval(model, dataloader, criterion, estimate_size):
    model.eval()
    total_loss = 0
    total_error = 0
    correct = 0
    total = 0
    # use estimate_size * number of data points for evaluation
    with torch.no_grad():
        for data, target in dataloader:
            if estimate_size > np.random.rand():
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                error = torch.abs(output - target).sum()
                total_error += error.item()
                # correct if predicted greater than 0.5 and target is 1 or predicted less than 0.5 and target is 0
                correct += ((output > 0.5) == (target > 0.5)).sum().item()
                total += target.size(0)
    n_samples = len(dataloader.dataset) * estimate_size
    avg_loss = total_loss / total
    avg_error = total_error / total
    avg_accuracy = correct / total
    model.train()
    return avg_loss, avg_error, avg_accuracy





def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = get_data_loader(batch_size)
    logging.info("Done loading data")
    ########################################################################
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    # optimizer = SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = Adam(net.parameters(), lr = learning_rate)
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
    print('neural net device', next(net.parameters()).device)
    estimate_size = 1.1
    # train_err[epoch], train_loss[epoch], train_acc[epoch] = eval(net, train_loader, criterion, estimate_size)
    val_err[epoch], val_loss[epoch],val_acc[epoch] = eval(net, val_loader, criterion, estimate_size)
    train_err[epoch], train_loss[epoch], train_acc[epoch] = val_err[epoch], val_loss[epoch], val_acc[epoch]
    
    print('neural net device', next(net.parameters()).device)
    logging.info(("Before training, Epoch {}: Train err: {}, Train loss: {}, Train acc: {} \n"+
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
    logging.info("Starting training...")
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
            total_train_loss += loss.item() * inputs.size(0)
            total_epoch += inputs.size(0)
            error = torch.abs(outputs - labels).sum()
            total_train_err += error.item()
            loss.backward()
            optimizer.step()    
       
        train_err[epoch], train_loss[epoch], train_acc[epoch] = total_train_err / total_epoch, total_train_loss / total_epoch, 1 - total_train_err / total_epoch
        # train_err[epoch], train_loss[epoch], train_acc[epoch] = eval(net, train_loader, criterion, estimate_size)
        val_err[epoch], val_loss[epoch],val_acc[epoch] = eval(net, val_loader, criterion, estimate_size)
        logging.info(("Epoch {}: Train err: {}, Train loss: {}, Train acc: {} \n"+
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
    logging.info('Finished Training')
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

import torch.nn.functional as F
class ChessNet2(nn.Module):
    def __init__(self):
        super(ChessNet2, self).__init__()
        self.name = "ChessNet2"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=17, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x



if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    ds = ChessDataset()
    ds.__getitem__(10000)[1]
    print('label_data: ', ds.__getitem__(10000)[1])


    # # load model from file
    # neuralnet = ChessNet2()
    # neuralnet.load_state_dict(torch.load('model_ChessNet2_bs1000_lr0.001_epoch50'))
    # neuralnet.to(device)
    ## good parameters: batch_size = 100, learning_rate = 0.01, num_epochs = 20


    batch_size = 10000
    learning_rate = 0.001
    num_epochs = 30

    neuralnet = ChessNet2()
    neuralnet.to(device)


    # print number of trainable parameters
    torchsummary.summary(neuralnet,(17,8,8))

    

    train_net(neuralnet, num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate)
    small_model_path = get_model_name(neuralnet.name, batch_size=batch_size, learning_rate=learning_rate, epoch=num_epochs)
    plot_training_curve(small_model_path)

    # print(sample input, sample outputs)
    sample_in = (ds.__getitem__(0)[0]).unsqueeze(0).to(device)
    print('sample_in: ', sample_in[0][6])
    sample_in[0][6][6][5] = 0.0
    sample_in[0][6][6][6] = 1.0
    print('sample_in: ', sample_in[0][6])

    sample_out = neuralnet(sample_in)
    print(sample_out)