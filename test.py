from torchvision import models,transforms,datasets
import torch
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import copy
from torch.optim import lr_scheduler
import io
from PIL import Image
import sys

use_gpu = torch.cuda.is_available()
#if (use_gpu): print ('GPU: True')
#else : print('GPU: False')
#print ('Torch version : ', torch.__version__)

class_names = ['gossiping', 'isolation', 'laughing', 'nonbullying', 'pullinghair', 'punching', 'quarrel', 'slapping', 'stabbing', 'strangle']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pretrained model from pytorch
cgf = [64, 64, 'M', 
       128, 128, 'M', 
       256, 256, 256, 'M', 
       512, 512, 512, 'M',
       512, 512, 512, 'M']

class BullyNet(nn.Module):
 
    def __init__(self, cfg, inchannels, num_classes = 10, init_weights= True, batch_norm= True):
        super(BullyNet, self).__init__()
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(inchannels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                inchannels = v
        
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        layers += [conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(1024*7*7, 4096),
            nn.ReLU(True), # inplace=True
            nn.Dropout(),
            nn.Linear(4096,256),
            nn.ReLU(True),
            nn.Dropout(), # p=0.5,inplace=False
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Dropout(), # p=0.5,inplace=False
            nn.Linear(128,num_classes),
        )
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
		

model_vgg = BullyNet(cgf, inchannels=3, num_classes=len(class_names))


#print(model_vgg.classifier[6].out_features) # 1000 
# Freeze training for all layers
for param in model_vgg.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
# num_features = model_vgg.classifier[6].in_features
# features = list(model_vgg.classifier.children())[:-1] # Remove last layer
# features.extend([nn.Linear(num_features, 10)]) # Add our layer with 4 outputs
# model_vgg.classifier = nn.Sequential(*features) # Replace the model classifier
# print(model_vgg)

#model_vgg = model_vgg.cuda()

model_vgg.load_state_dict(torch.load('VGG16_3_6.pt'))
model_vgg.eval()

img = Image.open(sys.argv[1])  # Read bytes and store as an img.


transform_pipeline = transforms.Compose([transforms.Resize(256),
										 transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
img = transform_pipeline(img)
img = img.unsqueeze(0)
img = Variable(img)

prediction = model_vgg(img)
index = prediction.data.numpy().argmax()  # Our prediction will be the index of the class label with the largest value.
print(class_names[index])