import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchsummary import summary
from torch.distributions import MultivariateNormal as MN


#define the model
class VariationalAE(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        super(VariationalAE, self).__init__()
        self.input_shape = input_shape
        self.conv_outs = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim


        self._num_conv_layers = len(conv_filters)


        #encoder
        self.encoder = nn.ModuleList()

        last_index = self.input_shape[0]
        for layer_index in range(self._num_conv_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=last_index,
                          out_channels=self.conv_outs[layer_index],
                          kernel_size=self.conv_kernels[layer_index],
                          stride=self.conv_strides[layer_index],
                          padding=(1,1)
                          ),
                nn.ReLU(),
                nn.BatchNorm2d(self.conv_outs[layer_index])
            )
            last_index = self.conv_outs[layer_index]
            self.encoder.append(layer)


        #bottle_neck
        self.bottle_neck  = nn.Sequential(nn.Flatten())
        self.mu = nn.Linear(64*7*7,self.latent_space_dim)
        self.log_variance = nn.Linear(64*7*7,self.latent_space_dim)

        #decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Sequential(nn.Linear(self.latent_space_dim, 64*7*7),
                                     nn.Unflatten(1, (64,7,7))))

        last_index = 64
        for layer_index in reversed(range(1,self._num_conv_layers)):
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels = last_index,
                          out_channels = self.conv_outs[layer_index],
                          kernel_size = self.conv_kernels[layer_index],
                          stride = self.conv_strides[layer_index],
                          padding = (1,1),
                          output_padding = self.conv_strides[layer_index]-1
                          ),
                nn.ReLU(),
                nn.BatchNorm2d(self.conv_outs[layer_index])
            )
            last_index = self.conv_outs[layer_index]
            self.decoder.append(layer)

        decoder_output = nn.Sequential(
            nn.ConvTranspose2d(in_channels = last_index,
                      out_channels = 1,
                      kernel_size = self.conv_kernels[0],
                      stride = self.conv_strides[0],
                      padding = (1,1),
                      output_padding = 0),
            nn.Sigmoid(),
            )
        self.decoder.append(decoder_output)




        # # x = nn.Flatten()(x)
        # x = nn.Linear(x.numel(),self.latent_space_dim)


        # print(self.encoder[self._num_conv_layers - 1][0].weight)

        # print(self.encoder)


    def forward(self,x):
        for l in self.encoder:
            x = l(x)
        mu,log_variance = self.bottleneck(x)
#
#         def pt_from_ND(mu, log_variance):
#             epsilon = torch.normal(mean=0., std=1., shape=(2,))
#             sampled_point = mu + torch.exp(log_variance/2) * epsilon
#             return sampled_point
#
        epsilon = torch.rand_like(log_variance)
        x = mu + log_variance*epsilon
        for i,l in enumerate(self.decoder):
            x = l(x)
        return x,log_variance,mu

    def bottleneck(self,x):
        x = self.bottle_neck(x)
        mu, log_variance = self.mu(x), self.log_variance(x)
        return mu,log_variance


if __name__ == "__main__":
    x = torch.randn(32,1,28,28)

    model = VariationalAE(input_shape=(1, 28, 28),
                          conv_filters=(32, 64, 64, 64),
                          conv_kernels=(3, 3, 3, 3),
                          conv_strides=(1, 2, 2, 1),
                          latent_space_dim=2)
    x,mu,sigma = model(x)
    print(x.shape, mu.shape, sigma.shape)
