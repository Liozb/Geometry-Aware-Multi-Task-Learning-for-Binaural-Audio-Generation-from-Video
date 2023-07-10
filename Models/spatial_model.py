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
        channel1_spec = input['channel1_spec'].to(device)
        channel2_spec = input['channel2_spec'].to(device)
        l = channel2_spec.shape[0]
        shape = (channel2_spec.shape[0],channel2_spec.shape[1]*2,channel2_spec.shape[2],channel2_spec.shape[3])
        audio = torch.zeros(shape).cuda()
        c = torch.zeros(l).cuda()
        for i in range(l):
            chanel1 = channel1_spec[i,:,:,:]
            chanel2 = channel2_spec[i,:,:,:]

            if np.random.random() < 0.5:
                cl_spec = torch.cat((chanel1, chanel2), dim=0)
                label = torch.FloatTensor([0])
            else:
                cl_spec = torch.cat((chanel2, chanel1), dim=0)
                label = torch.FloatTensor([1])
            audio[i,:,:,:] = cl_spec
            c[i] = label
            
        c_pred = self.net_audio(audio, visual_feature, self.name)
        # need a fix with passing through a classifier
        output = {'c_pred': c_pred, 'c': c}                     
        return output
    
if __name__ == "__main__":
    c1 = torch.randn([32, 2, 257, 64])
    c2 = torch.randn([32, 2, 257, 64])
    spec = {'channel1_spec': c1, 'channel2_spec': c2}
    Visualfeat = torch.randn([32,512,7,14])
    net = AudioNet(input_nc=4)
    model = modelSpatial(net)
    y = model(spec, Visualfeat)
    