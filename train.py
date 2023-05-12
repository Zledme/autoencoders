from autoencoder import Autoencoders
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_autoencoder(model, optimzer, loss_fn, timestamp, writer):
    for epoch in range(epochs):
        print("EPOCH {}".format(epoch + 1))

        model.train(True)

        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(trainloader):
            inputs, labels = data.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs,inputs)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(trainloader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        # model.train(False)
        #
        # running_vloss = 0.0
        # for i, vdata in enumerate(testloader):
        #     vinputs, vlabels = vdata
        #     voutputs = model(vinputs)
        #     vloss = loss_fn(voutputs, vinputs)
        #     running_loss += vloss
        #
        #     avg_vloss = running_vloss / (i + 1)
        # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        #
        # # Log the running loss averaged per batch
        # # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch + 1)
        # writer.flush()
        #
        # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp)
        #     torch.save(model.state_dict(), model_path)
    return model


def load_model():
    model = Autoencoders(
            input_shape=(1, 28, 28),
            conv_filters=(32, 64, 64, 64),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(1, 2, 2, 1),
            latent_space_dim=2
        ).to(device)
    return model

def load_data(batch_size = 32):
    # Define a transform to convert PIL images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    # Download and load the training data
    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Download and load the test data
    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)


    return trainloader, testloader

def others(lr = 0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer  = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    return optimizer, loss_fn, timestamp, writer


