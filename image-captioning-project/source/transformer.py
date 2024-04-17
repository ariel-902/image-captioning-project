import torch
from torch import nn, Tensor
import torch.nn.functional as F
import timm
from decoder import * 
import collections
import math
# from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, variant='vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True): # vit_base_patch16_224
    # def __init__(self, variant='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=pretrained)
        self.linear = nn.Linear(1024, 768)

    def forward(self, x):

        x = self.model.forward_features(x)
        x = self.linear(x)

        return x

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder()

        self.decoder = Decoder(config)

    def forward(self, img, caption_encoded):
        
        # print(img.size())
        x = self.encoder(img)
        # print(x.size())
        x = self.decoder(caption_encoded, x)
        # print(x.size())

        return x 


