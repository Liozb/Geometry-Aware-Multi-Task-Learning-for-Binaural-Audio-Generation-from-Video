# this file responsible for the spatial coherence train model

import random
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from params import *
from imports import *
from networks.Networks import *



class modelSpatial(torch.nn.Module):
    def __init__(self, audio_net):
        super(modelSpatial, self).__init__()
        self.net_audio = audio_net
        self.name = "spatial"

    def forward(self, input, visual_feature):
        cl_spec = input['cl_spec'].to(device) 
        label = input['label'].to(device) 
        
        pred = self.net_audio(cl_spec, visual_feature, self.name)
        output = {'cl_pred': pred, 'label': label}                     
        return output
    
if __name__ == "__main__":
    c1 = torch.randn([32, 2, 257, 64])
    c2 = torch.randn([32, 2, 257, 64])
    spec = {'channel1_spec': c1, 'channel2_spec': c2}
    Visualfeat = torch.randn([32,512,7,14])
    net = AudioNet(input_nc=4)
    model = modelSpatial(net)
    y = model(spec, Visualfeat)
    