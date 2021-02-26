import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms

from graphs.weights_initializer import weights_init

class GenerativeFSL_CAEModel(nn.Module):
    
    def __init__(self):
        super(GenerativeFSL_CAEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1, padding=1),  # b, 32, 224, 224
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 32, 112, 112
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1),  # b, 64, 112, 112
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 64, 56, 56
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # b, 128, 56, 56
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None)  # b, 128, 28, 28
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),  # b, 64, 56, 56
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),  # b, 32, 112, 112
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2,output_padding=1),  # b, 16, 224, 224
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),  # b, 3, 224, 224
					  #nn.Tanh()
        )
                # initialize weights
        self.apply(weights_init)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
