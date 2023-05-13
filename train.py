from autoencoder import Autoencoders
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

class Trainer:
    def __init__(self,epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = None
        self.loss_fn = None
        self.timestamp = None
        self.writer = None
        self.trainloader = None
        self.trainloader = None
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):

        self.load_data()
        self.load_model()
        self.others()
        for epoch in range(self.epochs):
            print("EPOCH {}".format(epoch + 1))
            self.model.train(True)

            running_loss = 0.
            last_loss = 0.

            pbar = tqdm(self.trainloader)
            for i,batch in enumerate(pbar):
                inputs, labels = batch

                inputs = inputs.to(self.device).type(torch.cuda.FloatTensor)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss_fn(outputs,inputs)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    tb_x = epoch * len(self.trainloader) + i + 1
                    self.writer.add_scalar('Loss/train', last_loss, tb_x)
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


    def load_model(self):
        self.model = Autoencoders(input_shape=(1, 28, 28),
                             conv_filters=(32, 64, 64, 64),
                             conv_kernels=(3, 3, 3, 3),
                             conv_strides=(1, 2, 2, 1),
                             latent_space_dim=2)
        self.model = self.model.to(self.device)

    def load_data(self):
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

        self.trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=self.batch_size,
                                              shuffle=True, num_workers=2)

        # Download and load the test data
        testset = torchvision.datasets.MNIST(root='./data',
                                             train=False,
                                             download=True,
                                             transform=transform)

        self.testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=2)


    def others(self):
        self.optimizer = optim.Adam(self.model.parameters(),
                                    self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer  = SummaryWriter('runs/Mnist_trainer_{}'.format(self.timestamp))
