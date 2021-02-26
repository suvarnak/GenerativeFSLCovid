"""
 discriminator model
"""
import torch
import torch.nn as nn
import torchvision.models as models
import json
from easydict import EasyDict as edict
from graphs.weights_initializer import weights_init


class EncoderModel(nn.Module):
    def __init__(self,config):
        super(EncoderModel, self).__init__()
        self.config = config

        self.num_classes = self.config.num_classes

        self.progress = 0.0

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1, padding=1),  # b, 32, 224, 224
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 32, 112, 112
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1),  # b, 64, 112, 112
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 64, 56, 56
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # b, 128, 56, 56
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 128, 28, 28
				)
        self.linear_layers = nn.Sequential(		
						nn.Linear(2*self.config.image_size*self.config.image_size, out_features=128),
            nn.Linear(128, out_features=self.config.num_ways),

        )


    def forward(self, x):  
        #x = self.encoder(x)
        #print(x.size())
        #self.discriminator = nn.Sequential(self.encoder, self.fc())
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        #print(x.size())

        #x = x.view(1, -1) 
        #x = self.fc(x)
        return x

class ConceptDiscriminatorModel(torch.nn.Module): #new model
    def __init__(self, pretrained_model):
        super(ConceptDiscriminatorModel, self).__init__()
        self.new_model = nn.Sequential(
            nn.Linear(in_features=512, out_features=30))
        self.pretrained_model = pretrained_model

    def forward(self, x):
        x = self.pretrained_model(x)
        return x
