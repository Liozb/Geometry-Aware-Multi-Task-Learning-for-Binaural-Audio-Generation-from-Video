# this file responsible for the spatial coherence train model

import sys
import random
import os
from Models.backbone_model import *
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *
from Datasets.AudioVisualDataset import AudioVisualDataset
class modelSpatial(torch.nn.Module):
    def __init__(self, audio_net):
        super(modelSpatial, self).__init__()
        self.net_audio = audio_net
        self.name = "spatial"

    def forward(self, input, visual_feature, volatile=False):
        c = []
        audio = []
        for i, data in enumerate(input):
            chanel1 = data['chanel1_spec']
            chanel2 = data['chanel2_spec']
            x = random.randint(0,1)
            if x == 1:
                chanel1, chanel2 = chanel2, chanel1
                c.append(1)
            elif x == 0:
                chanel1, chanel2 = chanel1, chanel2
            audio.append([chanel1,chanel2])
        audio_spectrogram = Variable(audio, requires_grad=False, volatile=volatile)
        visualAudio = self.net_audio(audio_spectrogram, visual_feature, self.name)
        output = {'visualAudio': visualAudio, 'c': c}                     
        return output