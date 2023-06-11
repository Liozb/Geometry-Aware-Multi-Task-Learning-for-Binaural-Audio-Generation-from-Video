# this file responsible for the spatial coherence train model

import sys
import random
import os
from Models.backbone_model import *
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *


class modelSpatial(torch.nn.Module):
    def __init__(self, audio_net):
        super(modelSpatial, self).__init__()
        self.net_audio = audio_net
        self.name = "spatial"

    def forward(self, input, visual_feature):
        c = []
        audio = []
        channel1_spec = input['channel1_spec']
        channel2_spec = input['channel2_spec']
        for i in range(64):
            chanel1 = channel1_spec[i,:,:,:]
            chanel2 = channel2_spec[i,:,:,:]
            x = random.randint(0,1)
            if x == 1:
                chanel1, chanel2 = chanel2, chanel1
                c.append(1)
            elif x == 0:
                chanel1, chanel2 = chanel1, chanel2
                c.append(0)
            audio.append([chanel1,chanel2])

        c_pred = self.net_audio(audio, visual_feature, self.name)
        # need a fix with passing through a classifier 
        output = {'c_pred': c_pred, 'c': c}                     
        return output
    
if __name__ == "__main__":
    x = torch.randn(1, )
    net = modelSpatial()
    