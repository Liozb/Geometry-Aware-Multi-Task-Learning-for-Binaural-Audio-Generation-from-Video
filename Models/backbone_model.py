import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from params import *

def get_spectrogram(input_spectrogram, mask_prediction):
    spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
    spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
    binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

    return binaural_spectrogram


class modelBackbone(torch.nn.Module):
    def __init__(self, backbone_nets):
        super(modelBackbone, self).__init__()
        #initialize model
        self.net_audio, self.net_fusion = backbone_nets
        self.name = "backbone"

    def forward(self, input, visual_feature, volatile=False):
        audio_diff = input['audio_diff_spec'].to(device)
        audio_mix = input['audio_mix_spec'].to(device)
        audio_gt = Variable(audio_diff[:,:,:-1,:], requires_grad=False)  # discarding the last time frame of the spectrogram(why?)

        input_spectrogram = Variable(audio_mix, requires_grad=False, volatile=volatile)
        mask_prediction, upfeatures = self.net_audio(input_spectrogram, visual_feature, self.name)

        #complex masking to obtain the predicted spectrogram
        binaural_spectrogram = get_spectrogram(input_spectrogram, mask_prediction)
        
        # predicted channels
        pred_left_mask, pred_right_mask = self.net_fusion(visual_feature, upfeatures)
        left_spectrogram = get_spectrogram(input_spectrogram, pred_left_mask).cuda()
        right_spectrogram = get_spectrogram(input_spectrogram, pred_right_mask).cuda()

        output =  {'mask_prediction': mask_prediction, 'binaural_spectrogram': binaural_spectrogram, 'audio_gt': audio_gt, 'left_spectrogram': left_spectrogram, 'right_spectrogram': right_spectrogram}
        return output